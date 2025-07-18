#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 2025

@author: benhills
"""

import numpy as np
import xarray as xr # type: ignore
from .sar_functions import *
from .supplemental import *


def sar_main(dat, image, N_aperture=None, stride=1, n_subaps=11, d=2000.,
             compression_type='standard', adaptive_window='linear-doppler',
             **kwargs):
    """
    Performs image compression in the along-track (azimuth) dimension using various methods.
    The compression is done in the frequency domain and can be applied to a migrated image
    or a range-compressed image.

    dat : xarray.DataArray
        Input data array containing the SAR data.
    image : str
        The name of the image variable to be processed.
    N_aperture : int, optional
        Number of traces in the aperture for processing.
        Default is None, which uses the total number of traces.
    n_subaps : int, optional
        Number of sub-apertures for sub-aperture processing. Default is 1.
    compression_type : str, optional
        Type of compression to apply. Options are:
            - 'quick-look': Coherent stack along horizontal.
            - 'standard': Standard SAR focusing with a single nadir aperture.
            - 'subap': Sub-aperture processing.
            - 'adaptive': Adaptive windowing based on Doppler spectra.
        Default is 'standard'.
    adaptive_window : str, optional
        Type of adaptive window to use for 'adaptive' compression. Options are:
            - 'linear-doppler': Linear Doppler-based windowing.
            - 'fixed-bw': Adaptive Doppler centroid but fixed bandwidth
            - 'fully-adaptive': Adaptive Doppler centroid and adaptive bandwidth
        Default is 'linear-doppler'.
    approximate : bool, optional
        Whether to use approximate calculations for range. Default is True.

    Returns
    -------
    image_ac : xarray.DataArray
        The along-track compressed image.
    """

    print('Compressing along-track dimension...')

    ### Quick Look Compression ###

    if compression_type == 'quick-look':
        # quick look processing is simply the coherent stack along horizontal
        if 'slowtime' in dat.dims:
            dat = dat.coarsen(slowtime=N_aperture,
                               boundary='trim').mean()
        elif 'distance' in dat.dims:
            dat = dat.coarsen(distance=N_aperture,
                               boundary='trim').mean()

    ### Standard and Non-Adaptive Subapertures ###

    elif compression_type in ['standard', 'subap']:

        # reshape the data array into apertures
        dat = reshape_on_aperture(dat, **kwargs)

        # calculate the doppler spectra
        dat = xr.apply_ufunc(calculate_doppler_spectra, dat,
                            input_core_dims=[["doppler"]],
                            output_core_dims=[["doppler"]],
                            exclude_dims=set(("doppler",)),
                            keep_attrs=True)

        # get the reference function from expected range offset
        dat = get_reference_function(dat, **kwargs)

        if compression_type == 'subap':
            # reshape on squinted subapertures
            # Define the size of a subaperture
            N_subap = int(len(dat.doppler) // ((1 + n_subaps) / 2))
            # expand dimensions to include subapertures
            dat = dat.rolling(doppler=N_subap,
                                min_periods=1).construct('subap',
                                                        stride=N_subap//2,
                                                        fill_value=0.)
            # swap the dimension names to keep Doppler at the end
        dat = dat.rename({'subap':'doppler','doppler':'subap'})

        # use the reference function to focus the image
        dat = xr.apply_ufunc(focus, dat, dat['C_ref'],
                            input_core_dims=[["fasttime", "doppler"],["fasttime","doppler"]],
                            output_core_dims=[["fasttime","doppler"]], vectorize=True,
                            kwargs={'compression_type':compression_type,'n_subaps':n_subaps},
                            keep_attrs=True)

        # multilook on squinted subapertures
        if compression_type == 'subap':
            dat = dat.mean('subap')

        # cut away the aperture padding
        dat = dat.isel(doppler=np.arange(dat.N_aperture//2, 3*dat.N_aperture//2))
        # restack apertures into one image
        dat = dat.stack(alongtrack=("distance", "doppler"))

    ### Adaptive Subapertures ###

    elif compression_type == 'adaptive':
        # expand the array into a new dimension with a rolling window
        if not hasattr(dat,'doppler'):
            dat = dat.rolling(slowtime=N_aperture).construct('doppler',
                                                             stride=stride)
        # redefine the along-track positions to within the new rolling aperture
        xs = dat.dx*(np.arange(N_aperture) - N_aperture//2)

        # get the doppler spectra
        dfreq = xr.apply_ufunc(calculate_doppler_spectra,
                               dat[image],
                               input_core_dims=[["doppler"]],
                               output_core_dims=[["doppler"]],
                               exclude_dims=set(("doppler",)),
                               )

        if adaptive_window == 'linear-doppler':
            dat['image_ac'] = (('fasttime', 'slowtime'), dfreq.max('doppler').data)

        else:
            # get the reference function
            dat = get_reference_function(dat,xs)

            if adaptive_window == 'fixed-bw':
                adaptive_kwargs = {'ddopp': dat.avg_vel/dat.dx/N_aperture,
                                   'dopp_bw': kwargs.get('dopp_bw', 100.)}
            elif adaptive_window == 'fully-adaptive':
                adaptive_kwargs = {'ddopp': dat.avg_vel/dat.dx/N_aperture,
                                   'dopp_bw': None}

            # apply the adaptive squint function
            image_ac = xr.apply_ufunc(adaptive_squint,
                        dfreq,
                        dat['C_ref'],
                        input_core_dims=[['doppler'], ['doppler']],
                        output_core_dims=[[]],
                        kwargs=adaptive_kwargs,
                        vectorize=True)

    print('Compression finished.')

    return dat


def adaptive_squint(dopp_row, C_ref_fd_row, ddopp=7.74, dopp_bw=100.):
    """
    Computes the adaptive squint sample power by correlating a Doppler row
    with a reference frequency domain row within a specified bandwidth.

    Parameters:
        dopp_row (array-like): The Doppler row data.
        C_ref_fd_row (array-like): The reference frequency domain row data.
        ddopp (float, optional): The Doppler resolution. Default is 7.74.
        dopp_bw (float or str, optional): The Doppler bandwidth. If set to
            'adaptive', the bandwidth is determined adaptively. Default is 100.

    Returns:
        float: The computed sample power from the correlation.
    """

    # Doppler centroid
    dc = np.argmax(dopp_row)
    # window size
    if dopp_bw != 'adaptive':
        N_bw = np.ceil(dopp_bw/ddopp)
    else:
        N_bw = np.ceil(dopp_bw/ddopp)

    # define the extent to sample
    tnum = len(dopp_row)
    start, end = max(0, int(dc-N_bw//2)), min(tnum-1, int(dc+N_bw//2))

    # correlate in freqency space
    sample_power = np.sum(dopp_row[start:end]*C_ref_fd_row[start:end])

    return sample_power
