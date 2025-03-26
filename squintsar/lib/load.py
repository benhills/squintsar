#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2025

@author: benhills
"""

import numpy as np
import xarray as xr
from scipy.io import loadmat

"""
Load functions for the squintsar processing library
"""


def load_cresis_range_compressed(fn, img=0, dset=None, c=3e8, eps=3.15):
    """
    Load data from cresis matlab file

    Parameters
    ----------
    fn: str, file name
    img: int,
    """
    dat = loadmat(fn)
    # data image
    image = np.squeeze(dat['data'][0][img])
    # fast time
    fasttime = np.squeeze(dat['hdr'][0][0][13][0][img])
    dt = fasttime[1]-fasttime[0]
    # slow time
    slowtime = np.squeeze(dat['hdr']['gps_time'][0][0])
    slowtime -= slowtime[0]
    # geolocation
    lat = np.squeeze(dat['hdr']['records'][0][0][img][0][0][0][7])
    lon = np.squeeze(dat['hdr']['records'][0][0][img][0][0][0][8])

    # output as an xarray object
    if dset is None:
        dset = xr.Dataset({'image_pc': (['fasttime', 'slowtime'], image)},
                          coords={'fasttime': fasttime, 'slowtime': slowtime,
                                  'lon': ('slowtime', lon),
                                  'lat': ('slowtime', lat)},
                          attrs={'dt': dt,
                                 'c': c,        # free-space wave speed
                                 'eps': eps,    # permittivity
                                 'n': np.sqrt(eps)})

    # TODO: ad an option where the dataset already exists
    # but we want to load another DataArray
    # else:
    #    dset

    return dset
