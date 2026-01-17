# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Import des dépendances et utilitaires globaux (device, settings, tensorise, etc.)
from .Dependances import norms, crits, optims, rglen, device, pi, e, tensorise

# Modèle MLP principal + fonction d'entraînement associée
from .MLP import MLP

from .Trainer import Trainer

from .UI import *

__version__ = "0.2.4"