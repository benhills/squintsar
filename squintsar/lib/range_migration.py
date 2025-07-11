#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 2025

@author: benhills
"""

import numpy as np
import time
from numpy.fft import fft, ifft, fftshift
from .sar_geometry import snell, get_depth_dist, sar_raybend


def rm_main(dat, **kwargs):
    """
    Performs range migration on the input data based on expected Doppler frequencies 
    within a synthetic aperture (i.e. based on the trace spacing and the platform velocity).

    Parameters
    ----------
    dat : object
        Input data object containing the necessary variables for processing.
    **kwargs : dict
        Additional keyword arguments to be passed to the range migration functions.

    Returns
    -------
    image_mig : complex
        The migrated image after applying range migration.
    """

    print('Migrating...')
    start = time.time()

    # Determine the range to migrate
    ft_rm = rm_find_range(dat, **kwargs)

    # frequency shift the image
    if 'image_fd' in dat.variables:
        image_shift = fftshift(dat.image_fd, axes=1)
    else:
        image_fd = fft(dat.image_rc)
        image_shift = fftshift(image_fd, axes=1)

    # resample the data array onto the migrated sample numbers
    image_shift_mig = rm_resample(dat, image_shift, ft_rm)

    # undo the frequency shift
    image_fd_mig = fftshift(image_shift_mig, axes=1)

    # back to time domain
    image_mig = ifft(image_fd_mig)

    print('Migration finished in:', round(time.time()-start, 2), 'sec')

    return image_mig


def rm_find_range(dat, N_aperture=None, **kwargs):
    """
    Computes the range to migrate for a given doppler frequency and range bin.

    This function calculates the range offset for a synthetic aperture radar (SAR) 
    system by determining the expected range to a target based on squint angles 
    derived from Doppler frequencies.

    Parameters:
    -----------
    dat : object
        A data object containing SAR parameters and measurements. It must have 
        the following attributes:
        - tnum: Total number of time samples.
        - avg_vel: Average velocity of the platform.
        - dx: Spatial resolution in the x-direction.
        - c: Speed of light.
        - fc: Carrier frequency.
        - n: Refractive index.
        - snum: Number of slow-time samples.
        - fasttime: Array of fast-time data.
        - h: Height of the platform above the ground.
    N_aperture : int, optional
        Number of aperture samples to use. If not provided, defaults to `dat.tnum`.
    **kwargs : dict
        Additional keyword arguments passed to the `sar_raybend` function.

    Returns:
    --------
    ft_rm : numpy.ndarray
        A 2D array of shape (dat.snum, N_aperture) containing the calculated 
        range migration values for each slow-time sample and aperture position.
    """

    if N_aperture is None:
        N_aperture = dat.tnum

    # doppler bandwidth
    f_bw = dat.avg_vel/dat.dx
    # full doppler frequency band
    f_dopp = np.linspace(-f_bw/2., f_bw/2., N_aperture)

    # squint angles from doppler frequencies TODO: this may be insufficiently precise
    theta_sqs = np.arcsin(f_dopp*dat.c/(2.*dat.avg_vel*dat.fc)) # /dat.n)

    # calculate range offset from along-track distances
    ft_rm = np.zeros((dat.snum, N_aperture)).astype(float)
    for i, t0 in enumerate(dat.fasttime):

        d, _ = get_depth_dist(t0.data, dat.h)

        # get along-track distances from squint angles (air only)
        xs_air = dat.h*np.tan(theta_sqs)
        # for returns above the ice surface
        if d < 0:  
            # calculate expected delta range to target
            ft_rm[i] = (np.sqrt(dat.h**2.+(xs_air)**2.) - dat.h)/dat.c
        # for returns below the ice surface
        else:
            # get along-track distances from refracted squint angles (ice only)
            xs_ice = d*np.tan(snell(theta_sqs))  # TODO: this may be insufficiently precise
            # calculate expected range to target
            ft_rm[i] = sar_raybend(t0.data, dat.h, (xs_air+xs_ice), **kwargs)

    return ft_rm


def rm_resample(dat, image_shift, ft_rm):
    """
    Resamples a shifted image based on range migration indices.

    Parameters:
    dat : object
        An object containing the data properties. It must have the attributes:
        - `dt` (float): The time interval between samples.
        - `snum` (int): The total number of samples.
        - `tnum` (int): The total number of traces.
    image_shift : numpy.ndarray
        A 2D array representing the image (fft shifted) to be resampled. The shape
        should match the dimensions of `dat.snum` x `dat.tnum`.
    ft_rm : numpy.ndarray
        A 2D array representing the range migration values in time.

    Returns:
    numpy.ndarray
        A 2D array of the resampled image with the same shape as `image_shift`.
    """

    # convert to sample number and limit at total number of samples
    rm_sample = np.round(ft_rm/dat.dt).astype(int)
    # tile with the fasttime sample numbers
    rm_sample += np.transpose(np.tile(np.arange(dat.snum), (dat.tnum, 1)))
    # cannot resample beyond the end of the array
    rm_sample[rm_sample > dat.snum-1] = dat.snum-1

    # tile trace numbers into an array
    rm_trace = np.tile(np.arange(dat.tnum), (dat.snum, 1))

    # resample the shifted image at the given indices
    return image_shift[rm_sample, rm_trace]
