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
