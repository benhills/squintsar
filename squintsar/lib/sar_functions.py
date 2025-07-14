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
    dfreq_raw = np.fft.fftshift(image_fd, axes=-1)
    # Apply Hanning window to reduce spectral leakage
    hann = np.hanning(np.shape(image)[-1])
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
