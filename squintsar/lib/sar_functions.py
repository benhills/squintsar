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
