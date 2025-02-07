#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np


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


def get_doppler_centroid(image):
    """
    """

    nus, power = periodogram(image[si,max(0,ti-nn):min(tnum,ti+nn)], fs=1./Delta_az)
    nu_best = np.argsort(nus[1:])

    return nu_best


def dc_to_squint(dc,range,h,n=np.sqrt(3.15)):
    """
    Get squint angle from doppler centroid
    """

    return squint
