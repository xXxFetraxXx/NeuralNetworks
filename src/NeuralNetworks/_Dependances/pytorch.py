# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import platform

import torch
import torch.optim as optim
import torch.nn as nn

from torch.amp import autocast, GradScaler

PI2 = 2 * torch.pi


def get_best_device():
    os_name = platform.system().lower()

    # =========== APPLE SILICON (macOS) ===========
    if os_name == 'darwin':
        if torch.backends.mps.is_available():
            return torch.device('mps')

    # =========== WINDOWS ===========
    if os_name == 'windows':
        if torch.cuda.is_available():
            return torch.device('cuda')

    # =========== LINUX ===========
    if os_name == 'linux':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            return torch.device('cuda')
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')

    # =========== Unknown OS ===========
    return torch.device('cpu')


device = get_best_device()
dev = str(device)
scaler = GradScaler(dev)

# --- Optimisations CUDA ---
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
torch.backends.cudnn.allow_tf32 = True

# Paramètres autograd
torch._inductor.config.max_autotune = 'max'
torch.set_float32_matmul_precision('medium')


def tensorise(obj):
    return torch.as_tensor(obj, dtype=torch.float32)


class Container:
    def __init__(self, dictionnaire: dict):
        self.D = dictionnaire

    def __repr__(self):
        return "\n".join(self.D.keys())

    def get(self, name: str):
        value = self.D.get(str(name))
        if value is None:
            fall_back = list(self.D.keys())[0]
            print(f"Warning: '{str(name)}' not recognized, "
                  f"falling back to '{fall_back}'")
            value = self.D.get(fall_back)
        return value

    def add(self, name: str, value):
        """Ajoute une entrée au conteneur."""
        self.D[str(name)] = value


class StandardCriterion:
    """Adapte une loss PyTorch à la signature commune (net, inputs, outputs)."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, net, inputs, outputs):
        out = net.train_forward(inputs)
        return self.loss_fn(out, outputs).mean(dim=0)


class CompositeCriterion:
    """Combine plusieurs critères avec des poids.

    components : list de (criterion, weight)
    """

    def __init__(self, components: list):
        self.components = components

    def __call__(self, net, inputs, outputs):
        total = None
        for crit, weight in self.components:
            loss = crit(net, inputs, outputs)
            total = loss * weight if total is None else total + loss * weight
        return total


norms = Container({
    'Relu': nn.ReLU(),
    'LeakyRelu': nn.LeakyReLU(),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'GELU': nn.GELU(),
    'Mish': nn.Mish(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'Hardtanh': nn.Hardtanh(),
    'Softplus': nn.Softplus(),
    'Softsign': nn.Softsign(),
})

crits = Container({
    'MSE': StandardCriterion(nn.MSELoss(reduction='none')),
    'L1': StandardCriterion(nn.L1Loss(reduction='none')),
    'SmoothL1': StandardCriterion(nn.SmoothL1Loss(reduction='none')),
    'SoftMarginLoss': StandardCriterion(nn.SoftMarginLoss(reduction='none')),
    'Huber': StandardCriterion(nn.HuberLoss(reduction='none')),
    'CrossEntropy': StandardCriterion(nn.CrossEntropyLoss(reduction='none')),
    'KLDiv': StandardCriterion(nn.KLDivLoss(reduction='none')),
    'PoissonNLL': StandardCriterion(nn.PoissonNLLLoss(reduction='none')),
    'MultiLabelSoftMargin': StandardCriterion(
        nn.MultiLabelSoftMarginLoss(reduction='none')
    ),
})

optims = Container({
    'Adam': optim.Adam,
    'Adadelta': optim.Adadelta,
    'Adafactor': optim.Adafactor,
    'AdamW': optim.AdamW,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'NAdam': optim.NAdam,
    'RAdam': optim.RAdam,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop,
    'SGD': optim.SGD,
})

torch.cuda.empty_cache()
