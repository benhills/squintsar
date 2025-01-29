#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 2025

@author: benhills
"""

import numpy as np

"""
Geometric functions for the squintsar processing library
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


def get_depth(r0, h, theta, n=n):
    """
    Use Snell's law and simple trigonometry to find the depth to target.

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


def get_theta(x, h, d, n=n):
    """
    Use a small angle approximation to get the squint angle
    from known geometry.

    Parameters
    ----------
    x:  float,  along-track distance from instrument to target
    h:  float, height of instrument above ice surface
    d:  float,  depth of target beneath air-ice interface
    n:  float, refractive index of second material (first assumed air)

    Output
    ----------
    theta:  float, squint angle (propagation direction through air)
    """

    # TODO: this is an approximation which assumes small angles
    # the exact solution is viable to implement although slower
    return x/(h+d/n)


def get_range(x, h, theta, n=n):
    """
    Use a small angle approximation to get
    the squint angle from known geometry.

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

    # Snells law
    theta_ice = snell(theta, n)
    # propagation through air
    r_air, x_air = h/np.cos(theta), h*np.tan(theta)
    # along-track distance in ice
    x_ice = x - x_air
    # propagation through ice
    r_ice = x_ice/np.sin(theta_ice)

    return r_air+r_ice


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
    d, x0 = get_depth(r0, h, theta)

    # small offsets to the squint angle within the aperture
    thetas = get_theta(x-x0, h, d)

    # range within aperture
    r = get_range(x-x0, h, thetas)

    return r-r0
