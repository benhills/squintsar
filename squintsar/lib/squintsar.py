#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft
from sar_geometry import get_depth_dist, sar_raybend
from sar_functions import matched_filter
from supplemental import r2p

"""
"""


def sar_extent(t0, h, theta_sq, theta_beam=.1, dx=1):
    """
    Define the aperture extent based on the half beamwidth and squint angle.
    Convert to index of the image array.
    """

    # for a given squint angle (theta) find the depth in ice
    # and along-track distance (x) from center of aperture to target
    d, x0 = get_depth_dist(t0, h, theta_sq)
    # define the synthetic aperture extents
    d_, x_start = get_depth_dist(t0, h, theta_sq+theta_beam/2.)
    d_, x_end = get_depth_dist(t0, h, theta_sq-theta_beam/2.)

    # TODO: explain this
    x_start *= -1
    x_end *= -1

    # aperture extents (index)
    ind_start = np.round(x_start/(dx)).astype(int)
    ind_end = np.round(x_end/(dx)).astype(int)

    # along-track distance for all points in the synthetic aperture
    x_sa = np.linspace(x_start, x_end, (ind_end-ind_start)+1)

    return x_sa+x0, ind_start, ind_end


def fill_reference_array(ft, h, theta_sq, theta_beam=0.1, dx=1, c=3e8):
    """

    """

    d, x0 = get_depth_dist(max(ft), h, theta_sq)
    Xs, ind0max, ind_max = sar_extent(max(ft), h, theta_sq, theta_beam, dx=dx)
    if ind0max > 0:
        ind0max = 0
        Xs = np.arange(0, max(Xs)-x0, dx) + x0
    if ind_max < 0:
        ind_max = 0
        Xs = np.arange(min(Xs)-x0, 0, dx) + x0
    n_x_max = ind_max-ind0max

    C_ref_all = np.zeros((len(ft), n_x_max+1), dtype=np.complex128)
    for si, ti in enumerate(ft):
        # get aperture extents
        x, ind0, ind_ = sar_extent(ti, min(h, ti*c), theta_sq, theta_beam, dx=dx)

        # calculate range and reference function
        r = sar_raybend(ti, min(h, ti*c), x, theta_sq)
        C_ref = matched_filter(r2p(r))
        # place in oversized array
        C_ref_all[si, ind0-ind0max:ind_-ind0max+1] = C_ref

    return C_ref_all


def range_migration():
    """

    """
    return


def sar_compression(image_rc, C_ref, ind_max, ind0max, domain='time'):
    """

    """
    # perform one correlation to determine the length of the output
    tnum_ = np.correlate(image_rc[0], C_ref[0], mode='full').shape[0]
    # initialize an expanded array
    snum = np.shape(image_rc)[0]
    image_rcac = np.zeros((snum, tnum_), dtype=np.complex128)

    # do the correlation (faster in frequency domain)
    for si in range(snum):
        if domain == 'time':
            image_rcac[si] = np.correlate(image_rc[si], C_ref[si], mode='full')
        elif domain == 'freq':
            C_ref_c = np.conjugate(C_ref[si]) # TODO: need to expand this array
            N = -int(C_ref.shape[1]/2)
            image_fq = fft(image_rc[si])
            C_ref_fq = fft(np.roll(C_ref_c, N))
            image_rcac[si] = ifft(image_fq*C_ref_fq)

    # crop down to the original size
    image_rcac_crop = image_rcac[:, ind_max:np.shape(image_rcac)[1]+ind0max]

    return image_rcac_crop
