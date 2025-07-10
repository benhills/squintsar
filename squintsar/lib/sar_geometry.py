#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 2025

@author: benhills

Geometric functions for the squintsar processing library
Following Heliere et al. (2007)
https://doi.org/10.1109/TGRS.2007.897433
"""

import numpy as np


def snell(theta, n=np.sqrt(3.15)):
    """
    This function calculates the refracted angle based on Snell's Law, which describes 
    the relationship between the angles of incidence and refraction when a wave passes 
    through the interface between two media with different refractive indices.

    theta : float
        Squint angle (propagation direction through air), in radians.
    n : float
        Refractive index of the second material (e.g., ice).

    Returns
    -------
    float
        The refracted angle, in radians.
    """
    # refraction at air-ice interface
    return np.arcsin(np.sin(theta)/n)


def get_range(x, h, d, s, n=np.sqrt(3.15), c=3e8):
    """
    Calculate the delta range to a target in seconds.

    This function computes the time it takes for a signal to travel to a target and back, 
    accounting for propagation through air and a second material with a specified refractive index.
    This is given as a difference from the range at the center of the aperture.

    x : float
        Along-track distance from the instrument to the target (in meters).
    h : float
        Height of the instrument above the ice surface (in meters).
    d : float
        Depth of the target below the ice surface (in meters).
    s : float
        Horizontal offset of the target from the point of refraction (in meters).
    n : float, optional
        Refractive index of the second material (default is the refractive index of ice, `n`).
    c : float, optional
        Speed of light in vacuum (default is 3e8 m/s).

    Returns
    -------
    float
        Range to the target, in seconds, minus the range at the center of the aperture (i.e. returns deltaR)
        accounting for propagation through air and the second material.
    """

    # propagation through air
    delta_r_air = np.sqrt(h**2.+(x-s)**2.) - h
    # propagation through ice
    delta_r_ice = np.sqrt(d**2.+s**2.) - d

    return delta_r_air/c + delta_r_ice*n/c


def get_depth_dist(t0, h, theta=0, n=np.sqrt(3.15), c=3e8):
    """
    Calculate the depth of a target beneath the air-ice interface and the along-track 
    distance to the closest approach using Snell's law and trigonometry.

    t0 : float
        Measured range to the target (in seconds).
    h : float
        Height of the instrument above the ice surface (in meters).
    theta : float, optional
        Squint angle, representing the propagation direction through air (in radians). 
        Default is 0.
    n : float
        Refractive index of the second material. Default is for ice. 
        The first material is assumed to be air.
    c : float, optional
        Speed of light in a vacuum (in meters per second). Default is 3e8.

    Returns
    -------
    d : float
        Depth of the target beneath the air-ice interface (in meters).
    x : float
        Along-track distance from the instrument to the target (in meters).
    """

    # Snells law
    theta_ice = snell(theta, n)
    # propagation through air
    r_air, x_air = h/np.cos(theta), h*np.tan(theta)
    # total propagation range along ray path
    r_ice = (t0 - r_air/c)*c/n
    # propagation through ice
    d, x_ice = r_ice*np.cos(theta_ice), r_ice*np.sin(theta_ice)

    return d, x_air+x_ice


def sar_raybend(t0, h, x, theta=0., n=np.sqrt(3.15), c=3e8, approximate=True):
    """
    Calculate the SAR range offset across the full aperture considering ray bending in two mediums.

    This function computes the range to a target, accounting for ray bending due to the refractive index 
    difference between two mediums (e.g., air and ice). The refractive index of the second medium is 
    provided as input, and the first medium is assumed to be air.

    t0 : float
        Measured range to the target (in seconds).
    h : float
        Height of the instrument above the ice surface (in meters).
    x : float or array
        Measured along-track distance (in meters).
    theta : float, optional
        Squint angle in radians. Default is 0.
    n : float
        Refractive index of the second medium. Default is for ice.
    c : float, optional
        Speed of light in m/s. Default is 3e8.
    approximate : bool, optional
        If True, use an approximate method to calculate the refraction point. Default is True.

    Returns
    -------
    r : float
        Range to the target, offset versus the range at the aperture center.
    """

    # for a given squint angle (theta) find the depth in ice
    # and along-track distance (x) from center of aperture to target
    d, x0 = get_depth_dist(t0, h, theta)

    # for returns above the ice surface
    if d < 0:  
        r = (np.sqrt(h**2.+(x-x0)**2.) - h)/c
    # for returns below the ice surface
    else:
        # get the refraction point
        if approximate:
            s = (x-x0)/(h*n/d+1)
        else:
            s = get_refraction_point(x-x0, h, d, n)
        # range within aperture
        r = get_range(x-x0, h, d, s)

    return r


def get_refraction_point(x, h, d, n=np.sqrt(3.15)):
    """
    Get the refraction point from known geometry by solving a fourth-order polynomial.

    This function calculates the along-track location where a ray intersects the ice surface 
    based on the geometry of the system and the refractive index of the second material.

    x : float or array-like
        Along-track distance from the instrument to the target. If an array is provided, 
        the function computes the refraction point for each element.
    h : float
        Height of the instrument above the ice surface.
    d : float
        Depth of the target beneath the air-ice interface.
    n : float, optional
        Refractive index of the second material (default is for ice, n = sqrt(3.15)).

    Returns
    -------
    float or numpy.ndarray
        The along-track location(s) where the ray intersects the ice surface. If `x` is a scalar, 
        a single float is returned. If `x` is an array, a numpy array of the same shape is returned.
    """

    # Ensure x is an array for consistent processing
    x = np.asarray(x)

    # Initialize an empty array to hold the refraction points
    s = np.empty_like(x)
    for i, xi in enumerate(x):
        # Coefficients for the polynomial
        # from Heliere et al. (2007) eq. (8)
        a4 = n**2.-1.
        a3 = -2*a4*xi
        a2 = a4*xi**2.+(n*h)**2.-d**2.
        a1 = 2*d**2.*xi
        a0 = -(d*xi)**2.

        # Define the coefficients of the polynomial
        # in descending order of power (e.g., a4x^4 + a3x^3 + a2x^2 + a1x + a0)
        coefficients = [a4, a3, a2, a1, a0]

        # Calculate the roots
        roots = np.roots(coefficients)
        # the smallest is the one we want
        s[i] = roots[np.argmin(abs(roots))]

    if len(s) == 1:
        return s[0]
    else:
        return s
