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


def rm_main(dat, image, theta_sq=0):
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

    # Determine the range to migrate
    ft_rm = rm_find_range(dat, theta_sq)

    # frequency shift the image
    image_shift = fftshift(image, axes=1)

    # resample the data array onto the migrated sample numbers
    image_shift_mig = rm_resample(dat, image_shift, ft_rm)

    # undo the frequency shift
    image_mig = fftshift(image_shift_mig, axes=1)

    return image_mig


def rm_find_range(dat, theta_sq):
    """
    ### Determine range to migrate ###
    """

    # empty array to fill
    ft_rm = np.zeros((dat.snum, dat.tnum)).astype(float)

    for i, ti in enumerate(dat.fasttime.data):
        # sample from above the ice surface
        if ti < dat.h/dat.c:
            if i == 0:
                lam = dat.c/(dat.fc)
                f_doppler = get_doppler_freq(dat.tnum, theta_sq, dat.avg_vel,
                                             dat.dx, dat.fc, 1., dat.c)
                ra = None
            ft0 = ti*np.cos(theta_sq)
            ft_rm[i] = ft0*(1.-lam**2*f_doppler**2/(4.*dat.avg_vel**2))**(-.5)
        # sample from below the ice surface
        else:
            if ra is None:
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


def rm_resample(dat, image_shift, ft_rm):
    # convert to sample number and limit at total number of samples
    rm_sample = np.round((ft_rm-dat.fasttime[0].data)/(dat.dt)).astype(int)
    rm_sample[rm_sample > dat.snum-1] = dat.snum-1
    # tile trace numbers into an array
    rm_trace = np.tile(np.arange(dat.tnum), (dat.snum, 1))
    # resample the shifted image at the given indices
    return image_shift[rm_sample, rm_trace]
