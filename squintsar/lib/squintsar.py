#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft
from sar_geometry import get_depth_dist, sar_raybend, sar_extent
from sar_functions import matched_filter, get_doppler_freq
from supplemental import r2p


class squintsar():
    """
    Squinted SAR processing.

    Some additional articles for reference:
    Rodriguez et al. (2009) https://api.semanticscholar.org/CorpusID:17738554
    Ferro (2019) https://doi.org/10.1080/01431161.2019.1573339
    Heister and Scheiber (2018) https://doi.org/10.5194/tc-12-2969-2018
    """

    def __init__(self):
        self.c = 3e8         # free space wave speed
        self.eps = 3.15
        self.n = np.sqrt(self.eps)

    def sar_compression(self, ind_max, ind0max, rm_flag=False, theta_sq=0.):
        """

        """

        # measurements to frequency domain
        self.data = fft(self.data)

        # range migration
        if rm_flag:
            self.range_migration()

        # output image shape is same as input image
        data_ac = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        # loop through all range bins
        for si in range(self.snum):
            # TODO: come back to this for deeper understanding
            C_ref_c = np.zeros(self.tnum, dtype=np.complex128)
            # why was this flipped?
            # C_ref_c[-C_ref.shape[1]:] = np.conjugate(C_ref[si])
            C_ref_c[:self.C_ref.shape[1]] = np.conjugate(self.C_ref[si])
            N = int(ind0max)  # int(C_ref.shape[1]/2)
            C_ref_fq = fft(np.roll(C_ref_c, N))

            # correlate in freqency space
            data_ac[si] = self.data[si]*C_ref_fq

        # back to time domain
        self.data = ifft(data_ac)

        return

    def fill_reference_array(self, h, theta_sq):
        """

        """

        d, x0 = get_depth_dist(max(self.ft), h, theta_sq)
        Xs, ind0max, ind_max = sar_extent(max(self.ft), h, theta_sq,
                                          self.theta_beam, self.dx)
        if ind0max > 0:
            ind0max = 0
            Xs = np.arange(0, max(Xs)-x0, self.dx) + x0
        if ind_max < 0:
            ind_max = 0
            Xs = np.arange(min(Xs)-x0, 0, self.dx) + x0
        n_x_max = ind_max-ind0max

        self.C_ref = np.zeros((self.snum, n_x_max+1), dtype=np.complex128)
        for si, ti in enumerate(self.ft):
            # get aperture extents
            x, ind0, ind_ = sar_extent(ti, min(h, ti*self.c), theta_sq,
                                       self.theta_beam, self.dx)

            # calculate range and reference function
            r = sar_raybend(ti, min(h, ti*self.c), x, theta_sq)
            C_ = matched_filter(r2p(r, fc=self.fc))
            # place in oversized array
            self.C_ref[si, ind0-ind0max:ind_-ind0max+1] = C_

        return

    def range_migration(self, theta_sq=0.):
        """

        """

        # get expected frequency shifts (from doppler centroid)
        f_doppler = get_doppler_freq(self.tnum, theta_sq, self.v, self.dx)
        # range as a function of doppler frequency
        r0 = self.ft*self.c*np.cos(theta_sq)
        # grid the doppler frequencies with range
        F, R = np.meshgrid(f_doppler, r0)

        # range to migrate
        r_rm = (R*(1.-self.c**2*F**2/(4.*self.fc**2.*self.v**2))**(-.5))
        # convert to sample number
        r_rm_n = np.round((r_rm-r0[0])/(self.dt*self.c)).astype(int)

        # frequency shift the image
        self.data = np.fft.fftshift(self.data)

        # for each frequency, resample the data in range
        image_mig = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        for si in range(self.snum):
            for ti in range(self.tnum):
                r_ind = r_rm_n[si, ti]
                image_mig[si, ti] = self.data[min(r_ind, self.snum-1), ti]

        # undo the frequency shift
        self.data = np.fft.fftshift(image_mig)

        return
