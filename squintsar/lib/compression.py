#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 2025

@author: benhills
"""

import time
import numpy as np
import xarray as xr # type: ignore
from numpy.fft import fft, ifft, fftshift
from .sar_functions import *


def sar_main(dat, image, N_aperture=None, stride=1, n_subaps=1,
             compression_type='standard',
             adaptive_window='linear-doppler',
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

    if compression_type == 'quick-look':
        # quick look processing is simply the coherent stack along horizontal
        if 'slowtime' in dat.dims:
            dat = dat.coarsen(slowtime=N_aperture,
                               boundary='trim').mean()
        elif 'distance' in dat.dims:
            dat = dat.coarsen(distance=N_aperture,
                               boundary='trim').mean()

    elif compression_type in ['standard', 'subap']:

        # get the reference function
        xs = np.arange(-dat.L_aperture/2., dat.L_aperture/2., 
                    dat.dx) + dat.dx/2.
        C_ref_fd = get_reference_function(dat, xs, **kwargs)
        dat['C_ref_fd'] = (('fasttime', 'doppler'), C_ref_fd)

        # get the doppler spectra
        dat = xr.apply_ufunc(calculate_doppler_spectra,
                            dat,
                            input_core_dims=[["doppler"]],
                            output_core_dims=[["doppler"]],
                            exclude_dims=set(("doppler",)))

        if compression_type == 'standard':
            # standard sar focusing with a single nadir aperture
            dat = standard(dat, image, **kwargs)

        elif compression_type == 'subap':
            dat = subapertures(dat, image, n_subaps, **kwargs)

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
                        dat['C_ref_fd'],
                        input_core_dims=[['doppler'], ['doppler']],
                        output_core_dims=[[]],
                        kwargs=adaptive_kwargs,
                        vectorize=True)

    print('Compression finished.')

    return dat


def standard(dat, image, **kwargs):
    """
    Perform standard image compression using frequency domain correlation.

    This function computes the autocorrelation of an image in the frequency
    domain by applying a reference function and a Hanning window.

    Args:
        dat: Input data object containing the image and associated metadata.
        image: Key or identifier for the image to be processed within `dat`.
        **kwargs: Additional keyword arguments passed to the reference function.

    Returns:
        numpy.ndarray: The autocorrelated image in the spatial domain.
    """

    # apply a hanning window
    image_fd_hann = fftshift(fft(dat[image]), axes=-1)*np.hanning(np.shape(dat[image])[1])

    # correlate in freqency space
    image_ac = ifft(fftshift(image_fd_hann, axes=-1)*C_ref_fd)
    dat['image_ac'] = (('fasttime', 'slowtime'), image_ac)

    return dat


def focus(dat, image, n_subaps, **kwargs):
    """
    Processes subapertures of input data to multiple images focused with different
    portions of the Doppler spectra.

    This function divides the input data into subapertures, applies a Hanning
    window to each subaperture, and performs correlation in the frequency
    domain using a reference function. The result is an array of focused
    images for each subaperture which may either be analyzed individually or
    eventually recombined with an incoherent average.

    Args:
        dat: Input data object containing the dataset. It is expected
            to have attributes such as `snum` and support indexing with `image`.
        image: Key or identifier for the image to be processed within `dat`.
        n_subaps: Number of subapertures to divide the data into.
        **kwargs: Additional keyword arguments passed to the reference function
            generator.

    Returns:
        np.ndarray: A 3D complex array of shape `(dat.snum, N_aperture, n_subaps)`
        containing the autocorrelated images for each subaperture.
    """

    if method == 'standard':
        Hwindow = np.hanning(dat.N_aperture)
    elif method == 'subap':
        # Define the aperture size for a subaperture
        N_subap = int(np.shape(dat[image])[-1] // ((1 + n_subaps) / 2))
        # same size for hanning window
        Hwindow = np.hanning(N_subap)
    
    if method == 'standard':
        # correlate in freqency space
        dat['image_ac'] = (('fasttime', 'doppler'),
                           ifft(fftshift(dat[image], axes=-1)*dat.C_ref_fd))
    elif method == 'subap':
        # shift the reference function to be subset in Doppler space
        C_fd_shift = fftshift(C_ref_fd, axes=-1)
        # pre-allocate output
        image_ac = np.empty((dat.snum, N_subap, n_subaps), dtype=complex)
        for i in range(n_subaps):
            # define the extent to sample
            start, end = i * N_subap// 2, (i + 2) * N_subap// 2
            data_sub = Hwindow * data_fd[:, start:end]
            C_sub = C_fd_shift[:, start:end]
            # correlate in frequency space
            image_ac[:, :, i] = ifft(fftshift(data_sub*C_sub, axes=-1))

        # expand dimensions to include subapertures
        if not hasattr(dat,'subap'):
            dat.expand_dims(dim={"subap":n_subaps, "slowtime_subap":N_subap})

        # assign the focused subaperture images to the dataset
        dat['image_subap_ac'] = (('fasttime', 'slowtime_subap', 'subap'), image_ac)
        # combine subapertures through incoherent average
        dat['image_ac'] = (('fasttime', 'slowtime_subap'), 
                        np.mean(abs(image_ac), axis=-1))

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
