# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import torch
from .pytorch import device

pi = torch.pi
e = torch.e

def rglen(liste : list):
    return range(len(liste))

def smoothstep (
    init_lr : float,
    xa      : float,
    n       : int):

    t = torch.linspace(0.0, 1.0, n, device = device)
    return init_lr + (xa - init_lr) * (6*t**5 - 15*t**4 + 10*t**3)