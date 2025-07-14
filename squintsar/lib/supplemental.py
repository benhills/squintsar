#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2025

@author: benhills

Supplemental functions for the squintsar processing library
"""

import numpy as np
from pyproj import Proj
from scipy.interpolate import interp1d


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

    return dist


def resample_alongtrack(dat, dx=None):
    """
    Resamples data along the track by interpolating to a uniform spacing 
    in the along-track distance and calculates the average velocity to be 
    used in later processing steps.

    Parameters:
    -----------
    dat : xarray.DataArray or xarray.Dataset
        Input data with a 'distance' coordinate and a 'slowtime' coordinate.
        The 'distance' coordinate is assumed to represent the along-track 
        distance, and 'slowtime' represents the time dimension.
    dx : float, optional
        Desired spacing in the along-track direction. If not provided, the 
        average spacing of the input data is used.

    Returns:
    --------
    xarray.DataArray or xarray.Dataset
        Resampled data with uniform spacing in the along-track distance. 
        The returned object includes updated attributes:
        - 'dx': The spacing used for resampling.
        - 'avg_vel': The calculated average velocity based on spacing and time.
        - 'tnum': The number of points in the resampled distance coordinate.
    """

    # make distance the primary dimension along track
    dat = dat.swap_dims({'slowtime':'distance'})

    if dx is None:
        # average spacing in along-track direction
        dx = np.mean(np.gradient(dat.distance))
    xf = dat.distance[-1]
    x0 = dat.distance[0]
    dist_new = np.arange(x0, xf, dx)
    # interpolate to uniform spacing in alont-track distance
    dat = dat.interp(distance=dist_new, method='nearest')

    # calculate the average velocity based on spacing and time
    dst = np.mean(np.gradient(dat.slowtime))
    v = dx/dst

    # reassign attributes that have changed
    dat = dat.assign_attrs({'dx': dx, 'avg_vel': v, 'tnum': len(dat.distance)})

    return dat