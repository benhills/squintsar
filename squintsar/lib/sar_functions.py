#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
from .sar_geometry import sar_raybend
from .supplemental import r2p


def calculate_doppler_spectra(image, pad=0):
    """
    Calculates the Doppler spectra of an image using the FFT (Fast Fourier Transform).

    Parameters:
    image (ndarray): The input image data, typically a 2D array.

    Returns:
    ndarray: The Doppler frequency spectrum after applying FFT and a Hanning window.
    """

    # FFT and shift to center zero frequency
    image_fd = np.fft.fft(image, n=np.shape(image)[-1]+pad)
    dfreq = np.fft.fftshift(image_fd, axes=-1)

    return dfreq


def get_reference_function(dat, 
                           xs=None, pad=None, **kwargs):
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

    if xs is None:
        xs = np.arange(-dat.L_aperture/2., dat.L_aperture/2., dat.dx) + dat.dx/2.
    if pad is None:
        pad = dat.N_aperture//2

    # pre-allocate reference function
    C_ref = np.empty((dat.snum, dat.N_aperture)).astype(complex)
    for i, t0 in enumerate(dat.fasttime):
        # calculate expected range to target
        r = sar_raybend(t0.data, dat.h, xs, **kwargs)
        # range to phase
        phi = r2p(r, fc=dat.fc)
        # calculate reference function from range
        # Phase to complex number for matched filter in along-track compression
        C_ref[i] = np.exp(-1j*phi)

    # pad the array with zeros
    C_ref = np.pad(C_ref,((0,0),(pad,pad)))

    # conjugate of the reference function in frequency domain
    C_ref_fd = fft(np.roll(np.conjugate(C_ref), -int((dat.N_aperture)//2+pad)))

    # assign to the data array
    dat['C_ref'] = (('fasttime', 'doppler'), fftshift(C_ref_fd, axes=-1))

    return dat


def focus(spectra, C_ref, **kwargs):
    """
    Perform standard image compression using frequency domain correlation.

    This function computes the autocorrelation of an image in the frequency
    domain by applying a reference function and a Hanning window.

    Args:
        spectra: Input Doppler spectra to be focused
        C_ref: reference function calculated from expected range offsets
        **kwargs: Additional keyword arguments passed to the reference function.

    Returns:
        numpy.ndarray: The autocorrelated image in the spatial domain.
    """

    # Hanning window
    Hwindow = np.hanning(np.shape(spectra)[-1])
    # correlate in frequency space
    spectra = fftshift(spectra*Hwindow*C_ref, axes=-1)
      
    # inverse Fourier transform back to time domain
    return ifft(spectra)