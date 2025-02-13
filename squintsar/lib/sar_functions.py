#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np

# refractive index for ice
n = np.sqrt(3.15)


def matched_filter(phi):
    """
    Phase to complex number for matched filter in along-track compression

    Parameters
    ----------
    phi:    float, phase

    Output
    ----------
    C:      complex, matched filter
    """
    # matched filter
    return np.exp(-1j*phi)


def squint2dc(theta_sq, v, fc=150e6, n=n, c=3e8):
    """
    Squint angle to frequency of Doppler centroid

    Parameters
    ----------
    theta_sq:   float, squint angle
    v:          float, platform velocity
    fc:         float, center frequency
    c:          float, vacuum wave speed

    Output
    ----------
    C:      complex, matched filter
    """
    # TODO: needs to be updated for two-medium sounding
    f_dc = 4*np.pi*fc/c * v * np.sin(theta_sq)

    return f_dc


def get_doppler_freq(tnum, theta_sq=0., v=0., dx=1.):
    """

    """

    # determine the range of frequencies in the spectrogram
    f_bw = v/dx
    f = np.linspace(-f_bw/2., f_bw/2., tnum)

    # find doppler frequency from squint angle
    f_dc = squint2dc(theta_sq, v)

    # calculate the ambiguity of each of the frequencies in the spectrogram
    # and add it to the frequency of each bin
    f_doppler = f + f_bw * np.round((f_dc-f)/f_bw)

    return f_doppler


def dc2squint(f_dc, r, h, n=n):
    """
    Get squint angle from doppler centroid
    """

    return


def data2dc(image):
    """
    Doppler centroid from data.

    Parameters
    ----------
    image:   2-d array, squint angle

    Output
    ----------
    f_dc:      complex, matched filter
    """

    nus, power = periodogram(image[si,max(0,ti-nn):min(tnum,ti+nn)], fs=1./Delta_az)
    nu_best = np.argsort(nus[1:])

    return nu_best
