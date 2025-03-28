#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft
from .sar_geometry import get_depth_dist, sar_raybend, sar_extent
from .sar_functions import matched_filter, get_doppler_freq
from .supplemental import r2p


class sqsar():
    """
    Squinted SAR processing.

    Some additional articles for reference:
    Rodriguez et al. (2009) https://api.semanticscholar.org/CorpusID:17738554
    Ferro (2019) https://doi.org/10.1080/01431161.2019.1573339
    Heister and Scheiber (2018) https://doi.org/10.5194/tc-12-2969-2018
    """

    def __init__(self):
        self.c = 3e8        # free space wave speed
        self.eps = 3.15     # permittivity
        self.n = np.sqrt(self.eps)  # index of refraction

    def sar_compression(self, ind0max, mig_flag=False, theta_sq=0.):
        """
        Image compression in the along-track (azimuth) dimension.
        Done in frequency domain.
        Includes range migration if flag is set to true.

        Parameters
        ----------
        ind0max:    int, number of traces to roll the array by
                        set from the reference array which was already filled.
        rm_flag:    bool, flag for range migration
        theta_sq:   float, squint angle

        Output
        ----------
        image_ac:   complex, along-track compressed image
        """
        # measurements to frequency domain
        image_fd = fft(self.image_rc)

        # range migration
        if mig_flag:
            print('Migrating...')
            image_fd = self.range_migration(image_fd, theta_sq)
            print('migration finished.')

        # output image shape is same as input image
        self.image_ac = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        # reference array to be extended the full length of image
        C_ref_c = np.zeros(self.tnum, dtype=np.complex128)

        print('Compressing...')
        # loop through all range bins
        for si in range(self.snum):
            # TODO: come back to this for deeper understanding
            C_ref_c[:] = 0j
            # why was this flipped?
            # C_ref_c[-C_ref.shape[1]:] = np.conjugate(C_ref[si])
            C_ref_c[:self.C_ref.shape[1]] = np.conjugate(self.C_ref[si])
            # TODO: something up with this roll value for negative squints
            N = int(ind0max)  # int(C_ref.shape[1]/2)
            C_ref_fq = fft(np.roll(C_ref_c, N))

            # correlate in freqency space
            self.image_ac[si] = image_fd[si]*C_ref_fq
        print('compression finished.')

        # back to time domain
        self.image_ac = ifft(self.image_ac)

        return

    def fill_reference_array(self, h, theta_sq, raybend_flag=True):
        """
        Fill an array with reference functions for each range bin.

        Parameters
        ----------
        h:          float, height of instrument platform above ice surface
        theta_sq:   float, squint angle

        Output
        ----------
        C_ref:      complex, reference function to be used for compression
        """
        # TODO: I think this function is too complicated
        # TODO: since I switched to freq domain only
        # TODO: break it out like CReSIS toolbox does
        print('Filling the reference array...')

        # get the geometry
        d, x0 = get_depth_dist(max(self.ft), h, theta_sq)
        # extent of array to be filled based on geometry
        Xs, ind0max, ind_max = sar_extent(max(self.ft), h, theta_sq,
                                          self.theta_beam, self.dx)
        # extent needs to include 0
        if ind0max > 0:
            ind0max = 0
            Xs = np.arange(0, max(Xs)-x0, self.dx) + x0
        if ind_max < 0:
            ind_max = 0
            Xs = np.arange(min(Xs)-x0, 0, self.dx) + x0

        # empty array to fill
        n_x_max = ind_max-ind0max
        self.C_ref = np.zeros((self.snum, n_x_max+1), dtype=np.complex128)
        # for each range bin
        for si, ti in enumerate(self.ft):
            # get aperture extents
            x, ind0, ind_ = sar_extent(ti, min(h, ti*self.c), theta_sq,
                                       self.theta_beam, self.dx)
            # calculate range and reference function
            if raybend_flag:
                r = sar_raybend(ti, min(h, ti*self.c), x, theta_sq)
            else:
                r0 = ti*self.c*np.cos(self.theta_sq)
                r = np.sqrt(r0**2.+(x-x0)**2.)/self.c
            C_ = matched_filter(r2p(r, fc=self.fc))
            # place in oversized array
            self.C_ref[si, ind0-ind0max:ind_-ind0max+1] = C_

        print('reference array filled.')

        return

    def range_migration(self, image, theta_sq=0.):
        """
        Range migration set by the expected Doppler frequencies
        calculated with the prescribed squint angle.

        Parameters
        ----------
        image:      complex, input image to be migrated
        theta_sq:   float, squint angle

        Output
        ----------
        image_mig:   complex, migrated image
        """
        # get expected frequency shifts (from doppler centroid)
        f_doppler = get_doppler_freq(self.tnum, theta_sq, self.v, self.dx,
                                     self.fc, self.n, self.c)
        # range as a function of doppler frequency
        #r0 = self.ft #*(self.c/self.n)  # *np.cos(theta_sq) not migrating enough? TODO
        # grid the doppler frequencies with fast time
        FQ, FT = np.meshgrid(f_doppler, self.ft)

        # wavelength
        lam = self.c/(self.fc*self.n)
        # range to migrate TODO: still approximating ray bending
        ft_rm = (FT*(1.-lam**2*FQ**2/(4.*self.v**2))**(-.5))
        # convert to sample number
        ft_rm_n = np.round((ft_rm-self.ft[0])/(self.dt)).astype(int) # TODO needs an 'n' on the bottom?

        # frequency shift the image
        image_shift = np.fft.fftshift(image)

        # for each frequency, resample the data in range
        image_mig = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        for si in range(self.snum):
            for ti in range(self.tnum):
                r_ind = r_rm_n[si, ti]
                image_mig[si, ti] = image_shift[min(r_ind, self.snum-1), ti]

        # undo the frequency shift
        return np.fft.fftshift(image_mig)
