#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills
"""

import numpy as np
from pyproj import Proj

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


def r2p(r, fc=190e6):
    """
    Convert range to phase

    Parameters
    ----------
    r:  float,  range
    """
    # phase
    return 4.*np.pi*fc*r


def calc_dist(long, lat, epsg='3031'):
    """
    Calculate along-track distance from x/y coordinates

    Parameters
    ----------
    long:  float,  longitude
    lat:  float,  latitude
    epsg: str
    """
    proj_stere = Proj('epsg:'+epsg)
    x, y = proj_stere(long, lat)

    dist = np.cumsum(np.sqrt((np.diff(x))**2.+(np.diff(y))**2.))
    dist = np.insert(dist, 0, 0.)
    dx = np.mean(np.gradient(dist))

    return dist, dx
