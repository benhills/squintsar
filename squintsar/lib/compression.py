#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft
from .sar_geometry import sar_raybend, sar_extent
from .supplemental import r2p


def sc_main(dat, image_fc, theta_sq=0., compression_type='standard'):
    """
    Image compression in the along-track (azimuth) dimension.
    Done in frequency domain.

    Parameters
    ----------
    h:          float, height of instrument platform above ice surface
    rm_flag:    bool, flag for range migration
    theta_sq:   float, squint angle

    Output
    ----------
    image_ac:   complex, along-track compressed image
    """

    return image_ac_fd



def standard():

    # output image shape is same as input image
    self.image_ac = np.zeros((self.snum, self.tnum), dtype=np.complex128)
    # reference array to be extended the full length of image
    C_ref_c = np.zeros(self.tnum, dtype=np.complex128)

    # loop through all range bins
    for si, ti in enumerate(self.ft):

        # get aperture extents
        x, ind0 = sar_extent(ti, min(h, ti*self.c), theta_sq,
                             self.theta_beam, self.dx)
        # calculate range and reference function
        r = sar_raybend(ti, min(h, ti*self.c), x, theta_sq)
        C_ref = matched_filter(r2p(r, fc=self.fc))

        C_ref_c[:len(C_ref)] = np.conjugate(C_ref)
        # TODO: something up with this roll value for negative squints
        # TODO: Rosen had it centered always where roll = len(C_ref)/2
        C_ref_fq = fft(np.roll(C_ref_c, ind0))

        # correlate in freqency space
        self.image_ac[si] = image_fd[si]*C_ref_fq



def subapertures():


def linear_doppler():


def adaptive():


