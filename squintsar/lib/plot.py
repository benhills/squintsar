#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 2025

@author: benhills
"""

import numpy as np
import matplotlib.pyplot as plt
from .supplemental import dB


def imshow(image, vmin=-70, vmax=-40, cmap='Greys_r'):
    """

    """

    fig, ax = plt.subplots(1, 1)

    ax.imshow(np.real(dB(image)),
              aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Sample number')


def imdopp(spectra, vmin=-70, vmax=-40, cmap='Greys_r'):
    """

    """

    fig, ax = plt.subplots(1, 1)

    ax.imshow(np.real(dB(spectra)),
              aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel('Doppler bin')
    ax.set_ylabel('Sample number')


def imref(C_ref, vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted'):
    """

    """

    fig, ax = plt.subplots(1, 1)

    ax.imshow(np.real(np.angle(C_ref)),
              aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel('Doppler bin')
    ax.set_ylabel('Sample number')


def imsubaps(images, vmin=-70, vmax=-40, cmap='Greys_r'):
    """

    """

    fig, axs = plt.subplots(1, 11, figsize=(12, 4))

    for i in range(11):
        ax = axs[i]
        ax.imshow(np.real(dB(images[:, :, i])),
                  vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        if i != 0:
            ax.tick_params(labelleft=False)
