# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .MLP import MLP
from .CNN import CNN
from .Trainer import Trainer
# Import des dépendances et utilitaires globaux (device, settings, tensorise, etc.)
from ._Dependances import norms, crits, optims, rglen, device, PI, E, tensorise
from ._UI import losses, learnings
from ._shared import Module

__version__ = "1.1.0"
__update_doc__ = True
__update_package__ = False
