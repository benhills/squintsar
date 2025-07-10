#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft
from .sar_geometry import sar_raybend
from .supplemental import r2p


def reshape_on_aperture(dat, N_aperture):
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

    if N_aperture is None:
        N_aperture = dat.tnum

    # move back to time domain
    if 'image_fd_mig' in dat.variables:
        dat['image_mig'] = (('fasttime', 'slowtime'),
                            np.fft.ifft(dat.image_fd_mig))
        dat.drop_vars('image_fd_mig')
    if 'image_fd' in dat.variables:
        dat.drop_vars('image_fd')

    # Reshape the dataset
    dat_reshaped = dat.coarsen(slowtime=N_aperture, boundary='trim').construct(
        slowtime=("alongtrack", "doppler"))
    # swap the dimensions for easier calculations on doppler images
    dat_reshaped = dat_reshaped.transpose('alongtrack', 'fasttime', 'doppler')
    # add the number of traces in an aperture for reuse later
    dat_reshaped = dat_reshaped.assign_attrs({'N_aperture': N_aperture})

    return dat_reshaped


def calculate_doppler_spectra(image):
    """
    Calculates the Doppler spectra of an image using the FFT (Fast Fourier Transform).

    Parameters:
    image (ndarray): The input image data, typically a 2D array.

    Returns:
    ndarray: The Doppler frequency spectrum after applying FFT and a Hanning window.
    """

    # FFT and shift to center zero frequency
    image_fd = np.fft.fft(image)
    dfreq_raw = np.fft.fftshift(image_fd, axes=2)
    # Apply Hanning window to reduce spectral leakage
    hann = np.hanning(np.shape(image)[2])
    return dfreq_raw*hann


def get_reference_function(dat, xs, **kwargs):
    """
    Generates a reference function for phase return as an aircraft passes a target.

    This function computes a reference function used for along-track compression in 
    SAR focusing. It calculates the expected range to the target, converts the range 
    to phase, and generates a complex reference function for matched filtering. 
    The result is returned in the frequency domain.

    Parameters:
        dat (object): An object containing SAR data and parameters. Expected attributes:
            - tnum (int): Default number of apertures if `N_aperture` is not provided.
            - snum (int): Number of samples in the slow-time dimension.
            - dx (float): Spatial resolution in the along-track direction.
            - fasttime (array-like): Array of fast-time values.
            - h (float): Aircraft altitude.
            - fc (float): Center frequency of the radar.
        xs (float): Cross-track positions of the target through the aperture.
        **kwargs: Additional keyword arguments passed to the `sar_raybend` function.

    Returns:
        numpy.ndarray: A 2D array of the reference function in the frequency domain, 
        with dimensions `(dat.snum, N_aperture)`.
    """

    # number of traces in the aperture
    N_aperture = len(xs)

    # pre-allocate reference function
    C_ref = np.empty((dat.snum, N_aperture)).astype(complex)
    for i, t0 in enumerate(dat.fasttime):
        # calculate expected range to target
        r = sar_raybend(t0.data, dat.h, xs, **kwargs)
        # range to phase
        phi = r2p(r, fc=dat.fc)
        # calculate reference function from range
        # Phase to complex number for matched filter in along-track compression
        C_ref[i] = np.exp(-1j*phi)

    # conjugate of the reference function in frequency domain
    C_ref_fd = fft(np.roll(np.conjugate(C_ref), -int(N_aperture//2)))

    return C_ref_fd