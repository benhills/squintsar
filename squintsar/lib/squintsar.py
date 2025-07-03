#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

from numpy.fft import fft, ifft
from .range_migration import rm_main
from .compression import sar_compression


def squintsar(dat,
              compression_type='standard',
              theta_sq=0.,
              migration=True,
              ):
    """

    """

    # range-compressed image to frequency domain
    image_fd = fft(dat.image_rc)

    # range migration
    if migration:
        print('Migrating...')
        image_mig = rm_main(dat, image_fd, theta_sq)
        print('migration finished.')

    # SAR compression
    print('Compressing...')
    image_ac_fd = sar_compression(dat, image_mig, theta_sq)
    print('compression finished.')

    # return to time domain
    dat.image_ac = ifft(image_ac_fd)
