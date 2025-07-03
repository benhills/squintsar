#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 2025

@author: benhills
"""

import numpy as np
from np.fft import fftshift
from .sar_geometry import get_depth_dist
from .sar_functions import get_doppler_freq


def rm_main(dat, image, theta_sq):
    """
    Range migration set by the expected Doppler frequencies
    calculated with the prescribed squint angle.

    Parameters
    ----------
    image:      complex, input image to be migrated
    theta_sq:   float, squint angle

    Output
    ----------
    image_mig:   complex, migrated image
    """

    ft_rm = rm_find_range(dat, image, theta_sq)

    # convert to sample number
    rm_sample = np.round((ft_rm-dat.fasttime[0].data)/(dat.dt)).astype(int)
    # ft_rm_n = np.transpose(np.transpose(ft_rm_n) +
    #                       (1.-np.cos(theta_sq)) *
    #                       np.arange(np.shape(ft_rm_n)[0])).astype(int)

    # frequency shift the image
    image_shift = np.fft.fftshift(image, axes=1)

    image_shift_mig = rm_resample(dat, image_shift, rm_sample)

    # undo the frequency shift
    image_mig = fftshift(image_shift_mig, axes=1)

    return image_mig


def rm_find_range(dat, image, theta_sq):
    """
    ### Determine range to migrate ###
    """

    # empty array to fill
    ft_rm = np.zeros_like(image).astype(float)
    for i in range(0, dat.snum):
        ti = dat.fasttime[i].data
        if ti < dat.h/dat.c:
            lam = dat.c/(dat.fc)
            f_doppler = get_doppler_freq(dat.tnum, theta_sq, dat.avg_vel,
                                         dat.dx, dat.fc, 1., dat.c)
            ft0 = ti*np.cos(theta_sq)
            ft_rm[i] = ft0*(1.-lam**2*f_doppler**2/(4.*dat.avg_vel**2))**(-.5)
        else:
            lam = dat.c/(dat.fc*dat.n)
            f_doppler = get_doppler_freq(dat.tnum, theta_sq, dat.avg_vel,
                                         dat.dx, dat.fc, dat.n, dat.c)
            ra = dat.h/np.cos(theta_sq)
            d, x0 = get_depth_dist(ti, dat.h, theta_sq, n=dat.n, c=dat.c)
            ri = d*(1.+(1./dat.n) *
                    (1.-(dat.h**2./ra**2.) -
                     lam**2*f_doppler**2/(4.*dat.avg_vel**2)))**(-.5)
            ft_rm[i] = (ra+ri*dat.n)/dat.c*np.cos(theta_sq)

    return ft_rm


def rm_resample(dat, image_shift, rm_sample):
    snum = dat.snum
    # for each frequency, resample the data in range
    image_shift_mig = np.zeros((dat.snum, dat.tnum), dtype=np.complex128)
    for si in range(dat.snum):
        for ti in range(dat.tnum):
            ft_ind = rm_sample[si, ti]
            image_shift_mig[si, ti] = image_shift[min(ft_ind, snum-1), ti]

    return image_shift_mig
