#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2025

@author: benhills
"""

import numpy as np
import xarray as xr
from scipy.io import loadmat
from .supplemental import calc_dist


def load_cresis_range_compressed(fn, img=0, dset=None, c=3e8, eps=3.15):
    """
    Load data from a CReSIS MATLAB file and return it as an xarray Dataset.

    This function loads radar data stored in MATLAB files, extracting
    information such as the radar image, geolocation, and metadata, and
    organizes it into an xarray Dataset for further analysis.

    fn : str
        File path to the MATLAB file.
    img : int, optional
        Index of the radar image to load. Default is 0.
    dset : xarray.Dataset, optional
        An existing xarray Dataset to which the loaded data can be added.
        If None, a new Dataset is created. Default is None.
    c : float, optional
        Speed of light in free space (m/s). Default is 3e8.
    eps : float, optional
        Relative permittivity of the medium. Default is 3.15.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the radar image, geolocation, and metadata.
    """
    dat = loadmat(fn)
    # data image
    image = np.squeeze(dat['data'][0][img])
    snum, tnum = np.shape(image)
    # fast time
    fasttime = np.squeeze(dat['hdr'][0][0][13][0][img])
    dt = fasttime[1]-fasttime[0]
    # slow time
    slowtime = np.squeeze(dat['hdr']['gps_time'][0][0])
    slowtime -= slowtime[0]
    # geolocation
    lat = np.squeeze(dat['hdr']['records'][0][0][img][0][0][0][7])
    lon = np.squeeze(dat['hdr']['records'][0][0][img][0][0][0][8])
    dist = calc_dist(lon, lat)
    # TODO: so ugly
    f0 = dat['hdr']['records'][0][0][img][0][0][0][1][0][0][12][0][0][5][0][0][2][0][0]
    f1 = dat['hdr']['records'][0][0][img][0][0][0][1][0][0][12][0][0][5][0][0][3][0][0]
    fc = np.mean([f0,f1])

    # output as an xarray object
    if dset is None:
        dset = xr.Dataset({'image_rc': (['fasttime', 'slowtime'], image)},
                          coords={'fasttime': fasttime, 
                                  'slowtime': slowtime,
                                  'lon': ('slowtime', lon),
                                  'lat': ('slowtime', lat),
                                  'distance': ('slowtime', dist)},
                          attrs={'c': c,        # free-space wave speed
                                 'eps': eps,    # permittivity
                                 'n': np.sqrt(eps),  # refractive index
                                 'snum': snum,  # number of samples
                                 'dt': dt,      # fast time step
                                 'tnum': tnum,  # number of traces
                                 'f0': f0,      # start frequency of chirp
                                 'f1': f1,      # end frequency of chirp
                                 'fc': fc})     # center frequency

    # TODO: add an option where the dataset already exists
    # but we want to load another DataArray
    # else:
    #    dset

    return dset
