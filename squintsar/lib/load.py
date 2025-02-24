#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2025

@author: benhills
"""

import numpy as np
from scipy.io import loadmat

"""
Load functions for the squintsar processing library
"""


def load_cresis_range_compressed(self, fn, img=0):
    """
    Load data from cresis matlab file

    Parameters
    ----------
    fn: str, file name
    img: int,
    """
    dat = loadmat(fn)
    # data image
    self.image_rc = np.squeeze(dat['data'][0][img])
    self.snum, self.tnum = np.shape(self.image_rc)
    # fast time
    self.ft = np.squeeze(dat['hdr'][0][0][13][0][img])
    self.dt = self.ft[1]-self.ft[0]
    # slow time
    slowtime = np.squeeze(dat['hdr']['gps_time'][0][0])
    self.st = slowtime - slowtime[0]
    # geolocation
    self.lat = dat['hdr']['records'][0][0][img][0][0][0][7]
    self.long = dat['hdr']['records'][0][0][img][0][0][0][8]

    return
