#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 2025

@author: benhills
"""

import numpy as np
from numpy.fft import fft, ifft
from .sar_geometry import sar_raybend, sar_extent, get_depth_dist
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

    def sar_compression(self, h, mig_flag=False, hann_flag=True, theta_sq=0.):
        """
        Image compression in the along-track (azimuth) dimension.
        Done in frequency domain.
        Includes range migration if flag is set to True.

        Parameters
        ----------
        h:          float, height of instrument platform above ice surface
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
            image_fd = self.range_migration(image_fd, h, theta_sq)
            print('migration finished.')

        # output image shape is same as input image
        self.image_ac = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        # reference array to be extended the full length of image
        C_ref_c = np.zeros(self.tnum, dtype=np.complex128)

        print('Compressing...')
        # loop through all range bins
        for si, ti in enumerate(self.ft):

            # get aperture extents
            x, ind0 = sar_extent(ti, min(h, ti*self.c), theta_sq,
                                 self.theta_beam, self.dx)
            # calculate range and reference function
            r = sar_raybend(ti, min(h, ti*self.c), x, theta_sq)
            C_ref = matched_filter(r2p(r, fc=self.fc))
            if hann_flag:
                C_ref *= np.hanning(len(C_ref))
            # place in the full array and roll to center it on low freq
            C_ref_c[:len(C_ref)] = np.conjugate(C_ref)
            C_ref_fq = fft(np.roll(C_ref_c, ind0))

            # correlate in freqency space
            self.image_ac[si] = image_fd[si]*C_ref_fq
        print('compression finished.')

        # back to time domain
        self.image_ac = ifft(self.image_ac)

        return

    def range_migration(self, image, h, theta_sq=0.):
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
        # range to migrate
        ft_rm = np.zeros_like(self.image_rc).astype(float)
        for i in range(0, self.snum):
            ti = self.ft[i]
            if ti < h/self.c:
                lam = self.c/(self.fc)
                f_doppler = get_doppler_freq(self.tnum, theta_sq, self.v,
                                             self.dx, self.fc, 1., self.c)
                ft0 = ti*np.cos(theta_sq)
                ft_rm[i] = ft0*(1.-lam**2*f_doppler**2/(4.*self.v**2))**(-.5)
            else:
                lam = self.c/(self.fc*self.n)
                f_doppler = get_doppler_freq(self.tnum, theta_sq, self.v,
                                             self.dx, self.fc, self.n, self.c)
                ra = h/np.cos(theta_sq)
                d, x0 = get_depth_dist(ti, h, theta_sq, n=self.n, c=self.c)
                ri = d*(1.+(1./self.n) *
                        (1.-(h**2./ra**2.) -
                         lam**2*f_doppler**2/(4.*self.v**2)))**(-.5)
                ft_rm[i] = (ra+ri*self.n)/self.c*np.cos(theta_sq)
        """
        # get expected frequency shifts (from doppler centroid)
        f_doppler = get_doppler_freq(self.tnum, theta_sq, self.v, self.dx,
                                     self.fc, self.n, self.c)
        # grid the doppler frequencies with fast time
        FQ, FT = np.meshgrid(f_doppler, self.ft)

        # wavelength
        lam = self.c/(self.fc*self.n)
        # range to migrate
        # TODO: fill in a loop
        ft_rm = (FT) * \
                (1.-lam**2*FQ**2/(4.*self.v**2))**(-.5)
        """
        # convert to sample number
        ft_rm_n = np.round((ft_rm-self.ft[0])/(self.dt)).astype(int)
        # ft_rm_n = np.transpose(np.transpose(ft_rm_n) +
        #                       (1.-np.cos(theta_sq)) *
        #                       np.arange(np.shape(ft_rm_n)[0])).astype(int)

        # frequency shift the image
        image_shift = np.fft.fftshift(image)

        # for each frequency, resample the data in range
        image_mig = np.zeros((self.snum, self.tnum), dtype=np.complex128)
        for si in range(self.snum):
            for ti in range(self.tnum):
                ft_ind = ft_rm_n[si, ti]
                image_mig[si, ti] = image_shift[max(min(ft_ind,
                                                        self.snum-1), 0), ti]

        # undo the frequency shift
        return np.fft.fftshift(image_mig)
