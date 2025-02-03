#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np

"""
Supplemental functions for the squintsar processing library
"""


def dB(P):
    """
    Convert power to decibels

    Parameters
    ----------
    P:  float,  power
    """

    return 10.*np.log10(P)


def r2p(r, fc=150e6, c=3e8):
    """
    Convert range to phase

    Parameters
    ----------
    r:  float,  range
    """

    # phase
    return 4.*np.pi*fc*r/c


def calc_dist(x, y):
    """
    Calculate along-track distance from x/y coordinates

    Parameters
    ----------
    x:  float,  x position
    y:  float,  y position
    """

    dist = np.cumsum(np.sqrt((np.diff(x))**2.+(np.diff(y))**2.))
    return np.insert(dist, 0, 0.)
