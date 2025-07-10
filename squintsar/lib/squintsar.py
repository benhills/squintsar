#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

from numpy.fft import fft, ifft
from .range_migration import rm_main
from .compression import sar_main


def squintsar(dat, migration=True, **kwargs):
    """
    Perform SAR (Synthetic Aperture Radar) image processing on the input data.

    This function applies azimuth compression and optionally range migration 
    to the input data, and returns the processed image in the time domain.

    Parameters:
        dat (object): Input data object containing the SAR image in range-compressed 
                      format (`dat.image_rc`).
        migration (bool, optional): Whether to perform range migration. Defaults to True.
    """

    # range-compressed image to frequency domain
    image_fd = fft(dat.image_rc)

    # range migration
    if migration:
        image_mig = rm_main(dat, image_fd, **kwargs)

    # SAR compression
    print('Compressing...')
    image_ac_fd = sar_main(dat, image_mig, **kwargs)
    print('compression finished.')

    # return to time domain
    dat.image_ac = ifft(image_ac_fd)

    return dat