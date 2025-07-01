#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np
from scipy.signal import periodogram
from .sar_geometry import snell


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


def squint2dc(theta_sq, v, fc=190e6, n=1.775, c=3e8):
    """
    Squint angle to frequency of Doppler centroid

    Parameters
    ----------
    theta_sq:   float, squint angle
    v:          float, platform velocity

    Output
    ----------
    f_dc:      float, frequency at Doppler centroid
    """
    lam = c/(fc)
    f_dc = 2. * v * np.sin(snell(theta_sq, n))/lam

    return f_dc


def get_doppler_freq(tnum, theta_sq=0., v=100., dx=1.,
                     fc=190e6, n=1.775, c=3e8):
    """
    Doppler frequencies for full length of array

    Parameters
    ----------
    tnum:       int, number of traces in array
    theta_sq:   float, squint angle
    v:          float, platform velocity
    dx:         float, spatial step between traces

    Output
    ----------
    f_doppler:  float, doppler frequencies across array
    """
    # doppler bandwidth
    f_bw = v/dx
    # range of frequencies in the spectrogram
    f = np.linspace(-f_bw/2., f_bw/2., tnum)

    # find doppler centroid from squint angle
    f_dc = squint2dc(theta_sq, v, fc, n, c)

    # find frequencies shift from squint
    # and add it to the frequency array
    f_doppler = f + f_bw * np.round((f_dc-f)/f_bw)

    return f_doppler


def dc2squint(f_dc, v, fc=190e6, n=1.775, c=3e8):
    """
    Get squint angle from doppler centroid

    Parameters
    ----------
    f_dc:      float, frequency at Doppler centroid
    v:         float, platform velocity

    Output
    ----------
    theta_sq:   float, squint angle
    """
    theta = np.arcsin(f_dc*c/(2.*v*fc))
    # inverse of snells law
    return snell(theta, 1./n)

def data2dc(image, v, dx=1.,
            fc=190e6, c=3e8):
    """
    Doppler centroid from data.

    Parameters
    ----------
    image:   2-d array
    dx:

    Output
    ----------
    f_dc:
    """
    # TODO: not tested
    snum, tnum = np.shape(image)
    P_dop = np.empty((0, tnum-1))
    for i in range(snum):
        nus, power = periodogram(image[i], fs=1./dx)
        idx = np.argsort(nus[1:])
        P_dop = np.append(P_dop, [power[1:][idx]], axis=0)

    # extract wavenumber with most power
    nus_full = nus[idx]
    nu_best = nus_full[np.argmax(P_dop, axis=1)]

    # convert to frequency and return
    return nu_best*c/(2.*v*fc)
