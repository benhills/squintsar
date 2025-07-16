#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills

Supplemental functions for the squintsar processing library
"""

import warnings
import numpy as np
from pyproj import Proj
from scipy.interpolate import interp1d


def dB(P):
    """
    Convert power to decibels.

    Parameters:
        P (float): The power value to be converted to decibels.

    Returns:
        float: The power value in decibels.
    """

    return 10.*np.log10(P)


def r2p(r, fc=190e6):
    """
    Convert range (in seconds) to phase.

    r : float
        Range in seconds.
    fc : float, optional
        Carrier frequency in Hz. Default is 190e6.

    Returns
    -------
    float
        Phase corresponding to the given range.
    """

    return 4.*np.pi*r*fc


def calc_dist(long, lat, epsg='3031'):
    """
    Calculate the along-track distance from longitude and latitude coordinates.

    This function projects geographic coordinates (longitude and latitude) into a
    specified EPSG coordinate system, computes the cumulative distance along the
    track, and calculates the mean spacing between points.

    long : array-like
        Longitude values in decimal degrees.
    lat : array-like
        Latitude values in decimal degrees.
    epsg : str, optional
        EPSG code for the projection system to use. Default is '3031'
        (Antarctic Polar Stereographic).

    Returns
    -------
    dist : numpy.ndarray
        Cumulative along-track distance in the projected coordinate system.
    dx : float
        Mean spacing between points along the track.
    """

    proj_stere = Proj('epsg:'+epsg)
    x, y = proj_stere(long, lat)

    dist = np.cumsum(np.sqrt((np.diff(x))**2.+(np.diff(y))**2.))
    dist = np.insert(dist, 0, 0.)

    return dist


def resample_alongtrack(dat, image, dist_new=None, dx=None, 
                        method='sinc', filt_len=None):
    """
    Resamples data along the track by interpolating to arbitrary spacing
    (defaults to uniform spacing) in the along-track distance and 
    calculates the average velocity to be used in later processing steps.

    Parameters:
    -----------
    dat : xarray.DataArray or xarray.Dataset
        Input data with a 'distance' coordinate and a 'slowtime' coordinate.
        The 'distance' coordinate is assumed to represent the along-track
        distance, and 'slowtime' represents the time dimension.
    dist_new: 1D numpy array of desired output sample positions
    dx : float, optional
        Desired spacing in the along-track direction. If not provided, the
        average spacing of the input data is used.
    method : string, optional
    filt_len : int, optional

    Returns:
    --------
    xarray.DataArray or xarray.Dataset
        Resampled data with uniform spacing in the along-track distance.
        The returned object includes updated attributes:
        - 'dx': The spacing used for resampling.
        - 'avg_vel': The calculated average velocity based on spacing and time.
        - 'tnum': The number of points in the resampled distance coordinate.
    """

    if dist_new is None:
        if dx is None:
            # average spacing in along-track direction
            dx = np.mean(np.gradient(dat.distance))
        xf = dat.distance[-1]
        x0 = dat.distance[0]
        dist_new = np.arange(x0, xf, dx)

    if filt_len is None:
        filt_len = dx*16

    data_in = dat[image].data.copy()
    data_out = np.zeros((dat.snum, len(dist_new))).astype(complex)

    if method == 'sinc':
        start_idx = np.searchsorted(dat.distance, dist_new[0] - filt_len/2, 
                                    side='right') - 1
        stop_idx = np.searchsorted(dat.distance, dist_new[0] + filt_len/2, 
                                side='right') - 1
        start_idx = max(start_idx, 0)
        stop_idx = max(stop_idx, 0)

        for ti, xi in enumerate(dist_new):
            while start_idx < dat.tnum and dat.distance[start_idx] < xi - filt_len/2:
                start_idx += 1
            while stop_idx < dat.tnum and dat.distance[stop_idx] < xi + filt_len/2:
                stop_idx += 1

            idxs = np.arange(start_idx, stop_idx)
            if len(idxs) == 0:
                data_out[:, ti] = 0
            else:
                x_off = dat.distance[idxs].data - xi
                # Compute windowed sinc function
                Hwin = (0.5 + 0.5 * np.cos(2 * np.pi * x_off / filt_len)) * \
                    np.sinc(x_off / dx).astype(complex)
                
                if len(Hwin) == 1:
                    data_out[:, ti] = data_in[:, idxs[0]]
                else:
                    # Calculate normalization factor norm_Hwin
                    # Differences between scaled x_off values
                    diffs = np.zeros_like(x_off)
                    if len(x_off) > 2:
                        diffs[0] = x_off[1] - x_off[0]
                        diffs[1:-1] = (x_off[2:] - x_off[:-2]) / 2
                        diffs[-1] = x_off[-1] - x_off[-2]
                    elif len(x_off) == 2:
                        diffs = np.array([x_off[1] - x_off[0], x_off[1] - x_off[0]])
                
                    norm_Hwin = np.minimum(dx, np.abs(diffs)) / dx
                    Hwin *= norm_Hwin
                    
                    # Weighted sum of input data
                    data_out[:, ti] = np.dot(data_in[:, idxs], Hwin)

    # make distance the primary dimension along track
    dat = dat.swap_dims({'slowtime':'distance'})
    # interpolate to uniform spacing in alont-track distance
    dat = dat.interp(distance=dist_new, method='linear')
    # calculate the average velocity based on spacing and time
    dst = np.mean(np.gradient(dat.slowtime))
    v = dx/dst
    # reassign attributes that have changed
    dat = dat.assign_attrs({'dx': dx, 'avg_vel': v, 'tnum': len(dat.distance)})

    dat[image].data = data_out

    return dat


def reshape_on_aperture(dat, d=1000., N_aperture=None):
    """
    Reshapes a dataset along the aperture dimension for SAR processing.

    Parameters:
    -----------
    dat : xarray.Dataset
        The input dataset containing SAR data. It is expected to have variables such as
        'image_fd_mig' or 'image_fd' for frequency domain data, and dimensions like
        'fasttime' and 'slowtime'.
    N_aperture : int or None
        The number of traces in an aperture. If None, it defaults to `dat.tnum`.

    Returns:
    --------
    xarray.Dataset
        The reshaped dataset with dimensions adjusted for aperture processing. The dataset
        will have the following modifications:
        - Frequency domain variables ('image_fd_mig', 'image_fd') are dropped or converted
          back to the time domain.
        - The 'slowtime' dimension is coarsened and split into 'alongtrack' and 'doppler'.
        - Dimensions are transposed for easier Doppler image calculations.
        - An attribute 'N_aperture' is added to the dataset for reuse later.
    """

    # calculate the synthetic aperture length based on geometry
    L_aperture = dat.c/dat.fc*(dat.h+d/dat.n)/(2*dat.dx)
    if N_aperture is None:
        N_aperture = int(L_aperture/dat.dx)
    elif N_aperture*dat.dx > L_aperture:
        warnings.warn('Aperture too large based on trace spacing, \
        automatically reducing N_aperture')
        N_aperture = int(L_aperture/dat.dx)

    # move back to time domain
    if 'image_fd_mig' in dat.variables:
        dat['image_mig'] = (('fasttime', 'distance'),
                            np.fft.ifft(dat.image_fd_mig))
        dat.drop_vars('image_fd_mig')
    if 'image_fd' in dat.variables:
        dat.drop_vars('image_fd')

    # Reshape the dataset
    dat_reshaped = dat.coarsen(distance=N_aperture, boundary='trim').construct(
        distance=("alongtrack", "doppler"))
    # swap the dimensions for easier calculations on doppler images
    dat_reshaped = dat_reshaped.transpose('alongtrack', 'fasttime', 'doppler')
    # add the number of traces in an aperture and aperture length for reuse later
    dat_reshaped = dat_reshaped.assign_attrs({'N_aperture': N_aperture, 
                                              'L_aperture': N_aperture*dat.dx})

    return dat_reshaped