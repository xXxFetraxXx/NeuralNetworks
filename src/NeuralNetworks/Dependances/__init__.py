# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .matplot import *
from .pytorch import *

import numpy as np
from PIL import Image

import copy
import subprocess
import requests
from io import BytesIO
from tqdm import tqdm
import plotly.graph_objects as go
from IPython.display import display, clear_output

from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split

import math
pi = math.pi
e = math.e

norms = lambda: print("""
"Relu"
"LeakyRelu"
"ELU"
"SELU"
"GELU"
"Mish"
"Sigmoid"
"Tanh"
"Hardtanh"
"Softplus"
"Softsign"
"""
)

crits = lambda: print("""
"MSE"
"L1"
"SmoothL1"
"Huber"
"CrossEntropy"
"KLDiv"
"PoissonNLL"
"MultiLabelSoftMargin"
"""
)

optims = lambda: print("""
"Adadelta"
"Adafactor"
"Adam"
"AdamW"
"Adamax"
"ASGD"
"NAdam"
"RAdam"
"RMSprop"
"Rprop"
"SGD"
"""
)

def rglen(list):
    return range(len(list))

def fPrintDoc(obj):
    return lambda: print(obj.__doc__)