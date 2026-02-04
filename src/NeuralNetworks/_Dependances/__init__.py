# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator

plt.rcParams['figure.facecolor'] = (0,0,0,0)
plt.rcParams['axes.facecolor']   = (0,0,0,0)
plt.rcParams['axes.grid']  = True

from .pytorch import *
from .tools import *

import numpy as np

from tqdm.auto import trange, tqdm


import onnx
from onnxsim import simplify