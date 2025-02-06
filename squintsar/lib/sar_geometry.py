#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 2025

@author: benhills
"""

import numpy as np

"""
Geometric functions for the squintsar processing library
Following Heliere et al. (2007)
https://doi.org/10.1109/TGRS.2007.897433
"""

# refractive index for ice
n = np.sqrt(3.15)


def snell(theta, n=n):
    """
    Snell's Law

    Parameters
    ----------
    theta:  float, squint angle (propagation direction through air)
    n:      float, refractive index of second material (first assumed air)

    Output
    ----------
    float, refracted angle
    """

    # refraction at air-ice interface
    return np.arcsin(np.sin(theta)/n)


def get_depth_dist(r0, h, theta, n=n):
    """
    Use Snell's law and simple trigonometry to find the depth of target
    and along-track distance to closest approach.

    Parameters
    ----------
    r0: float, total range to target
    h:  float, height of instrument above ice surface
    theta:  float, squint angle (propagation direction through air)
    n:  float, refractive index of second material (first assumed air)

    Output
    ----------
    d:  float,  depth of target beneath air-ice interface
    x:  float,  along-track distance from instrument to target
    """

    # Snells law
    theta_ice = snell(theta, n)
    # propagation through air
    r_air, x_air = h/np.cos(theta), h*np.tan(theta)
    # total propagation range along ray path
    r_ice = r0 - r_air
    # propagation through ice
    d, x_ice = r_ice*np.cos(theta_ice), r_ice*np.sin(theta_ice)

    return d, x_air+x_ice


def get_refraction_point(x, h, d, n=n):
    """
    Get the refraction point from known geometry.
    Solve a fourth order polynomial.

    Parameters
    ----------
    x:  float,  along-track distance from instrument to target
    h:  float, height of instrument above ice surface
    d:  float,  depth of target beneath air-ice interface
    n:  float, refractive index of second material (first assumed air)

    Output
    ----------
    s:  float, along-track location where the ray intersects the ice surface
    """

    a4 = n**2.-1.
    a3 = -2*a4*x
    a2 = a4*x**2.+(n*h)**2.-d**2.
    a1 = 2*d**2.*x
    a0 = -d**2.*x**2.

    # Define the coefficients of the polynomial
    # in descending order of power (e.g., ax^4 + bx^3 + cx^2 + dx + e)
    coefficients = [a4, a3, a2, a1, a0]

    # Calculate the roots
    roots = np.roots(coefficients)

    return roots[np.argmin(abs(roots))]


def get_range(x, h, d, s, n=n):
    """
    Range to target.

    Parameters
    ----------
    x:  float,  along-track distance from instrument to target
    h:  float, height of instrument above ice surface
    theta:  float, squint angle (propagation direction through air)
    n:  float, refractive index of second material (first assumed air)

    Output
    ----------
    r:  float,  range to target
    """

    # propagation through air
    r_air = np.sqrt(h**2.+(x-s)**2.) - h
    # propagation through ice
    r_ice = np.sqrt(d**2.+s**2.) - d

    return r_air+n*r_ice


def aperture_extent(r0, h, theta_sq, theta_beam=.1, dx=1, x_ov=1):
    """
    Define the aperture extent based on the half beamwidth and squint angle.
    Convert to index of the image array.
    """

    # for a given squint angle (theta) find the depth in ice
    # and along-track distance (x) from center of aperture to target
    d, x0 = get_depth_dist(r0, h, theta_sq)
    # define the synthetic aperture extents
    d_, x_start = get_depth_dist(r0, h, theta_sq-theta_beam/2.)
    d_, x_end = get_depth_dist(r0, h, theta_sq+theta_beam/2.)

    # aperture extents (index)
    ind_start = np.round(x_start/(dx/x_ov)).astype(int)
    ind_end = np.round(x_end/(dx/x_ov)).astype(int)

    # along-track distance for all points in the synthetic aperture
    x_sa = np.linspace(x_start, x_end, (ind_end-ind_start)+1)

    return x_sa, ind_start, ind_end


def SAR_aperture_raybend(r0, h, x, theta=0., n=n):
    """
    Ray bending for sounding in two mediums.
    Calculate the SAR range offset across the full aperture.
    The refractive index (and notation) are for ice
    but another material could be substituted.

    Parameters
    ----------
    r0: float,  measured range to target
    h:  float, height of instrument above ice surface
    x:  float or array, measured along-track distance
    theta:  float, squint angle
    n:  float, refractive index of second material (first assumed air)

    Output
    ----------
    r:  float,  range to target (offset versus r0 at aperture center)
    """

    # for a given squint angle (theta) find the depth in ice
    # and along-track distance (x) from center of aperture to target
    d, x0 = get_depth_dist(r0, h, theta)

    # reference function placed consistently in the oversized array
    if d < 0:  # for returns above the ice surface
        r = np.sqrt(h**2.+(x-x0)**2.)-h
    else:
        # small offsets to the squint angle within the aperture
        s = get_refraction_point(x-x0, h, d)
        # range within aperture
        r = get_range(x-x0, h, d, s)

    return r
