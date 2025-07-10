#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills

Supplemental functions for the squintsar processing library
"""

import numpy as np
from pyproj import Proj


def dB(P):
    """
    Convert power to decibels.

    Parameters:
        P (float): The power value to be converted to decibels.

    Returns:
        float: The power value in decibels.
    """
    
    return 10.*np.log10(P)


def r2p(r, fc=190e6):
    """
    Convert range (in seconds) to phase.

    r : float
        Range in seconds.
    fc : float, optional
        Carrier frequency in Hz. Default is 190e6.

    Returns
    -------
    float
        Phase corresponding to the given range.
    """

    return 4.*np.pi*r*fc


def calc_dist(long, lat, epsg='3031'):
    """
    Calculate the along-track distance from longitude and latitude coordinates.

    This function projects geographic coordinates (longitude and latitude) into a 
    specified EPSG coordinate system, computes the cumulative distance along the 
    track, and calculates the mean spacing between points.

    long : array-like
        Longitude values in decimal degrees.
    lat : array-like
        Latitude values in decimal degrees.
    epsg : str, optional
        EPSG code for the projection system to use. Default is '3031' 
        (Antarctic Polar Stereographic).

    Returns
    -------
    dist : numpy.ndarray
        Cumulative along-track distance in the projected coordinate system.
    dx : float
        Mean spacing between points along the track.
    """

    proj_stere = Proj('epsg:'+epsg)
    x, y = proj_stere(long, lat)

    dist = np.cumsum(np.sqrt((np.diff(x))**2.+(np.diff(y))**2.))
    dist = np.insert(dist, 0, 0.)
    dx = np.mean(np.gradient(dist))

    return dist, dx
