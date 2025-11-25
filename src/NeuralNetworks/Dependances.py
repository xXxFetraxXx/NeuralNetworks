# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.quantization as tq
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader

import visualtorch

from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchvision.transforms import ToTensor, Resize, Compose

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import platform
import copy
import math
import subprocess
import requests
from io import BytesIO
import rff
from tqdm import tqdm

from IPython.display import display, clear_output

torch.cuda.empty_cache()

# --- Device global ---
# Utilise GPU si disponible, sinon CPU

def get_best_device():
    """
    Détermine automatiquement le meilleur backend PyTorch disponible selon l'OS
    et les GPU présents sur la machine.

    La priorité dépend du système :

    macOS (Apple Silicon)
    ---------------------
    - Utilise Metal/MPS si disponible.
    - Sinon CPU.

    Windows
    -------
    - CUDA si une GPU Nvidia CUDA est détectée.
    - Sinon DirectML (AMD / Intel / Nvidia sans CUDA).
    - Sinon CPU.

    Linux
    -----
    - CUDA (Nvidia)
    - ROCm (AMD)
    - oneAPI / XPU (Intel GPU)
    - Sinon CPU.

    Retour
    ------
    torch.device
        Le device optimal pour exécuter PyTorch selon le matériel détecté.

    Notes
    -----
    - Sur ROCm, PyTorch expose l'alias "cuda", ce qui est normal.
    - DirectML permet un fallback GPU universel sous Windows.
    - Cette fonction ne lance aucun test de performance, elle se base uniquement
      sur les backends disponibles dans l'installation PyTorch.
    """

    os_name = platform.system().lower()

    # =========== APPLE SILICON (macOS) ===========
    if os_name == "darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
 
    # =========== WINDOWS ===========
    if os_name == "windows":
        # 1) CUDA
        if torch.cuda.is_available():
            return torch.device("cuda")

    # =========== LINUX ===========
    if os_name == "linux":
        # 1) CUDA (Nvidia)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # 2) ROCm (AMD)
        elif hasattr(torch.backends, "hip") and torch.backends.hip.is_available():
            return torch.device("cuda")

        # 3) Intel oneAPI / XPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")

    # =========== Unknown OS ===========
    return torch.device("cpu")

device = get_best_device()

def has_gcc():
    """Return True if GCC is installed and callable."""
    try:
        r = subprocess.run(
            ["gcc", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False

# --- Paramètres graphiques globaux ---
# Fond transparent et couleur gris uniforme
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

# --- Optimisations CUDA ---
# Accélération des convolutions et matmul
torch.backends.cudnn.benchmark = True           # optimise selon les tailles de tenseurs
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True    # autorise TF32 (plus rapide sur Ampere+)
torch.backends.cudnn.allow_tf32 = True

# Paramètres autograd
torch.autograd.set_detect_anomaly(False)        # pas d'analyse lourde
torch.autograd.profiler.profile(enabled=False)
torch.use_deterministic_algorithms(False)      # privilégie la performance à la reproductibilité stricte

torch._inductor.config.max_autotune = "max"     # config max pour Torch-Inductor

# Constantes
pi = math.pi
e = math.e

# --- Liste des normalisations/activations disponibles ---
norm_list = {
    "Relu": nn.ReLU(),
    "LeakyRelu": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "SELU": nn.SELU(),
    "GELU": nn.GELU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Hardtanh": nn.Hardtanh(),
    "PReLU": nn.PReLU(),
    "RReLU": nn.RReLU(),
    "Softplus": nn.Softplus(),
    "Softsign": nn.Softsign()
}
norms = lambda: print("""
"Relu"
"LeakyRelu"
"ELU"
"SELU"
"GELU"
"Sigmoid"
"Tanh"
"Hardtanh"
"PReLU"
"RReLU"
"Softplus"
"Softsign"
"""
)

# --- Liste des fonctions de perte disponibles ---
crit_list = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
    "SmoothL1": nn.SmoothL1Loss(),
    "Huber": nn.HuberLoss(),
    "CrossEntropy": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
    "BCEWithLogits": nn.BCEWithLogitsLoss(),
    "KLDiv": nn.KLDivLoss(),
    "PoissonNLL": nn.PoissonNLLLoss(),
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss(),
}
crits = lambda: print("""
"MSE"
"L1"
"SmoothL1"
"Huber"
"CrossEntropy"
"BCE"
"BCEWithLogits"
"KLDiv"
"PoissonNLL"
"MultiLabelSoftMargin"
"""
)

# --- Création d’un dictionnaire d’optimiseurs ---
def optim_list(self, learning_rate):
    """
    Renvoie un dictionnaire d’optimiseurs PyTorch pour le MLP donné.

    Paramètres
    ----------
    self : objet
        Objet contenant `self.model` à optimiser.
    learning_rate : float
        Taux d’apprentissage à appliquer aux optimisateurs.

    Retour
    ------
    dict
        Dictionnaire {nom_optimiseur : instance_optimiseur}.
    """
    return {
        "Adadelta": optim.Adadelta(self.model.parameters(), lr=learning_rate),
        "Adafactor": optim.Adafactor(self.model.parameters(), lr=learning_rate),
        "Adam": optim.Adam(self.model.parameters(), lr=learning_rate),
        "AdamW": optim.AdamW(self.model.parameters(), lr=learning_rate),
        "Adamax": optim.Adamax(self.model.parameters(), lr=learning_rate),
        "ASGD": optim.ASGD(self.model.parameters(), lr=learning_rate),
        "NAdam": optim.NAdam(self.model.parameters(), lr=learning_rate),
        "RAdam": optim.RAdam(self.model.parameters(), lr=learning_rate),
        "RMSprop": optim.RMSprop(self.model.parameters(), lr=learning_rate),
        "Rprop": optim.Rprop(self.model.parameters(), lr=learning_rate),
        "SGD": optim.SGD(self.model.parameters(), lr=learning_rate)
    }

optims_list = {
    "Adadelta": "",
    "Adafactor": "",
    "Adam": "",
    "AdamW": "",
    "Adamax": "",
    "ASGD": "",
    "NAdam": "",
    "RAdam": "",
    "RMSprop": "",
    "Rprop": "",
    "SGD": "",
}
    
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

# --- Fonctions utilitaires ---
def rglen(list):
    """
    Renvoie un range correspondant aux indices d’une liste.

    Paramètres
    ----------
    list : list-like
        Objet dont on souhaite obtenir les indices.

    Retour
    ------
    range
        Range Python de 0 à len(list)-1.
    """
    return range(len(list))

def tensorise(obj):
    """
    Convertit un objet en tenseur PyTorch float32 et l’envoie sur le device global.

    Paramètres
    ----------
    obj : array-like, list, np.ndarray, torch.Tensor
        Objet à convertir en tenseur.

    Retour
    ------
    torch.Tensor
        Tenseur float32 sur le device global (CPU ou GPU).

    Notes
    -----
    - Harmonise les types pour MLP et autres traitements PyTorch.
    - Assure que les données sont compatibles avec les opérations GPU/CPU.
    """
    return torch.as_tensor(obj, dtype=torch.float32, device=device)

def fPrintDoc(obj):
    """
    Crée une fonction anonyme qui affiche le docstring d'un objet.

    Paramètres
    ----------
    obj : object
        Tout objet Python possédant un attribut `__doc__`.

    Retour
    ------
    function
        Une fonction sans argument. Lorsqu'on l'appelle, elle affiche
        le docstring de l'objet passé.

    Exemple
    -------
    >>> def ma_fonction():
    ...     '''Ceci est le docstring.'''
    ...     pass
    >>> print_doc = fPrintDoc(ma_fonction)
    >>> print_doc()
    Ceci est le docstring.
    """
    return lambda: print(obj.__doc__)

torch.cuda.empty_cache()