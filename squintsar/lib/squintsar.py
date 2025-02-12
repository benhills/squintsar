#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft
from sar_geometry import get_depth_dist, sar_raybend
from sar_functions import matched_filter, squint2dc
from supplemental import r2p

"""
This is...

Please cite...

Some additional articles for reference:
Rodriguez et al. (2009) https://api.semanticscholar.org/CorpusID:17738554
Ferro (2019) https://doi.org/10.1080/01431161.2019.1573339
Heister and Scheiber (2018) https://doi.org/10.5194/tc-12-2969-2018
"""


def sar_compression(image_rc, C_ref, ind_max, ind0max, domain='time'):
    """

    """
    # number of range bins
    snum = np.shape(image_rc)[0]

    # do the correlation (faster in frequency domain)
    if domain == 'time':
        # perform one correlation to determine the length of the output
        tnum_ = np.correlate(image_rc[0], C_ref[0], mode='full').shape[0]
        # initialize an expanded array
        image_rcac = np.zeros((snum, tnum_), dtype=np.complex128)
        # loop through all range bins
        for si in range(snum):
            # correlate measurements with the reference array
            image_rcac[si] = np.correlate(image_rc[si], C_ref[si], mode='full')
        # crop down to the original size
        image_rcac = image_rcac[:, ind_max:np.shape(image_rcac)[1]+ind0max]

    elif domain == 'freq':
        # output image shape is same as input image
        tnum = np.shape(image_rc)[1]
        image_rcac = np.zeros((snum, tnum), dtype=np.complex128)
        # loop through all range bins
        for si in range(snum):
            # measurements in frequency domain
            image_fq = fft(image_rc[si])
            # TODO: add range migration

            # TODO: come back to this for deeper understanding
            C_ref_c = np.zeros(tnum, dtype=np.complex128)
            # why was this flipped?
            # C_ref_c[-C_ref.shape[1]:] = np.conjugate(C_ref[si])
            C_ref_c[:C_ref.shape[1]] = np.conjugate(C_ref[si])
            N = int(ind0max)  # int(C_ref.shape[1]/2)
            C_ref_fq = fft(np.roll(C_ref_c, N))
            # correlate in freqency space
            image_rcac[si] = ifft(image_fq*C_ref_fq)

    return image_rcac


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


def fill_reference_array(ft, h, theta_sq, theta_beam=0.1, dx=1.,
                         fc=150e6, c=3e8):
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
        x, ind0, ind_ = sar_extent(ti, min(h, ti*c),
                                   theta_sq, theta_beam, dx=dx)

        # calculate range and reference function
        r = sar_raybend(ti, min(h, ti*c), x, theta_sq)
        C_ref = matched_filter(r2p(r, fc=fc))
        # place in oversized array
        C_ref_all[si, ind0-ind0max:ind_-ind0max+1] = C_ref

    return C_ref_all


def find_freq_shift(tnum, theta_sq=0., v=0., dx=1.):
    """

    """

    # determine the range of frequencies in the spectrogram
    f_bw = v/dx
    f = np.linspace(-f_bw/2., f_bw/2., tnum)

    # find doppler frequency from squint angle
    f_dc = squint2dc(theta_sq, v)

    # calculate the ambiguity of each of the frequencies in the spectrogram
    f_amb = f_bw * np.round((f_dc-f)/f_bw)

    # and add it to the frequency of each bin
    return (f + f_amb)  # .astype(int)


def range_migration(image, fasttime, theta_sq=0., v=0., lam=0., dx=1., c=3e8):
    """

    """

    # get expected frequency shifts (from doppler centroid)
    f_doppler = find_freq_shift(np.shape(image)[1],
                                theta_sq=theta_sq, v=v, dx=dx)

    # range as a function of doppler frequency
    r0 = fasttime/c*np.cos(theta_sq)
    r_rm = (r0*(1.-lam**2*f_doppler**2/(4.*v**2))**(-.5)).astype(int)
    # range to number of samples (to migrate)
    dt = fasttime[1]-fasttime[0]
    r_rm_n = np.round((r_rm-fasttime[0])/dt).astype(int)

    # frequency shift the image
    image_shift = np.fft.fftshift(image)

    # for each frequency, resample the data in range
    snum, tnum = image.shape
    image_mig = np.zeros((snum, tnum), dtype=np.complex128)
    for si in range(snum):
        for ti in range(tnum):
            r_ind = r_rm_n[si, ti]
            image_mig[si, ti] = image_shift[min(r_ind, snum-1), ti]

    # undo the frequency shift
    image_out = np.fft.fftshift(image_mig)

    return image_out
