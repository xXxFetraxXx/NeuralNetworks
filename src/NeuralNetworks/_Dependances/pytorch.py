# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os

import platform

import torch
import torch.optim as optim
import torch.nn as nn

from torch.amp import autocast, GradScaler

import visualtorch

pi2 = 2 * torch.pi

def get_best_device ():

    os_name = platform.system ().lower ()

    # =========== APPLE SILICON (macOS) ===========
    if os_name == 'darwin':
        if torch.backends.mps.is_available ():
            return torch.device ('mps')

    # =========== WINDOWS ===========
    if os_name == 'windows':
        # 1) CUDA
        if torch.cuda.is_available ():
            return torch.device ('cuda')

    # =========== LINUX ===========
    if os_name == 'linux':
        # 1) CUDA (Nvidia)
        if torch.cuda.is_available ():
            return torch.device ('cuda')
        # 2) ROCm (AMD)
        elif hasattr (torch.backends, 'hip') and torch.backends.hip.is_available ():
            return torch.device ('cuda')

        # 3) Intel oneAPI / XPU
        elif hasattr (torch, 'xpu') and torch.xpu.is_available ():
            return torch.device ('xpu')

    # =========== Unknown OS ===========
    return torch.device ('cpu')
    
device = get_best_device (); dev = str (device)
scaler = GradScaler (dev)

# --- Optimisations CUDA ---
# Accélération des convolutions et matmul
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
    def __init__ (self, dictionnaire : dict):
        self.D = dictionnaire

    def __repr__ (self):
        return "\n".join(self.D.keys())

    def get (self, name : str):
        value = self.D.get (str(name))
        if value is None:
            fall_back = list(self.D.keys ())[0]
            print(f"Warning: '{str(name)}' not recognized, falling back to '{fall_back}'")
            value = self.D.get (fall_back)
        return value

norms = Container ({
    'Relu'     : nn.ReLU      (),
    'LeakyRelu': nn.LeakyReLU (),
    'ELU'      : nn.ELU       (),
    'SELU'     : nn.SELU      (),
    'GELU'     : nn.GELU      (),
    'Mish'     : nn.Mish      (),
    'Sigmoid'  : nn.Sigmoid   (),
    'Tanh'     : nn.Tanh      (),
    'Hardtanh' : nn.Hardtanh  (),
    'Softplus' : nn.Softplus  (),
    'Softsign' : nn.Softsign  ()
})

crits = Container ({
    'MSE'                  : nn.MSELoss                  (reduction='none'),
    'L1'                   : nn.L1Loss                   (reduction='none'),
    'SmoothL1'             : nn.SmoothL1Loss             (reduction='none'),
    'SoftMarginLoss'       : nn.SoftMarginLoss           (reduction='none'),
    'Huber'                : nn.HuberLoss                (reduction='none'),
    'CrossEntropy'         : nn.CrossEntropyLoss         (reduction='none'),
    'KLDiv'                : nn.KLDivLoss                (reduction='none'),
    'PoissonNLL'           : nn.PoissonNLLLoss           (reduction='none'),
    'MultiLabelSoftMargin' : nn.MultiLabelSoftMarginLoss (reduction='none')
})

optims = Container ({
    'Adam'      : optim.Adam     ,
    'Adadelta'  : optim.Adadelta ,
    'Adafactor' : optim.Adafactor,
    'AdamW'     : optim.AdamW    ,
    'Adamax'    : optim.Adamax   ,
    'ASGD'      : optim.ASGD     ,
    'NAdam'     : optim.NAdam    ,
    'RAdam'     : optim.RAdam    ,
    'RMSprop'   : optim.RMSprop  ,
    'Rprop'     : optim.Rprop    ,
    'SGD'       : optim.SGD
})

torch.cuda.empty_cache ()