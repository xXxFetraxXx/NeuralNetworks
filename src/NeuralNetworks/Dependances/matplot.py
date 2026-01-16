# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['figure.facecolor'] = (0,0,0,0)
plt.rcParams['axes.facecolor']   = (0,0,0,0)
grey_color = "#888888"

# Style général du texte et axes
plt.rcParams['text.color']       = grey_color
plt.rcParams['axes.labelcolor']  = grey_color
plt.rcParams['xtick.color']      = grey_color
plt.rcParams['ytick.color']      = grey_color
plt.rcParams['axes.edgecolor']   = grey_color
plt.rcParams['axes.titlecolor']  = grey_color

# Activation de la grille globale
plt.rcParams['axes.grid']  = True
plt.rcParams['grid.color'] = grey_color