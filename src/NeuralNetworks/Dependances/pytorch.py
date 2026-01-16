# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import platform

import torch
import torch.optim as optim
import torch.nn as nn
import torch.quantization as tq
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader

from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchvision.transforms import ToTensor, Resize, Compose

torch.cuda.empty_cache()
def get_best_device():

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

# --- Optimisations CUDA ---
# Accélération des convolutions et matmul
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paramètres autograd
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.use_deterministic_algorithms(False)

torch._inductor.config.max_autotune = "max"

norm_list = {
    "Relu": nn.ReLU(),
    "LeakyRelu": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "SELU": nn.SELU(),
    "GELU": nn.GELU(),
    "Mish": nn.Mish(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Hardtanh": nn.Hardtanh(),
    "Softplus": nn.Softplus(),
    "Softsign": nn.Softsign()
}

crit_list = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
    "SmoothL1": nn.SmoothL1Loss(),
    "Huber": nn.HuberLoss(),
    "CrossEntropy": nn.CrossEntropyLoss(),
    "KLDiv": nn.KLDivLoss(),
    "PoissonNLL": nn.PoissonNLLLoss(),
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss()
}

def optim_list(params):
    return {
        "Adadelta": optim.Adadelta(params),
        "Adafactor": optim.Adafactor(params),
        "Adam": optim.Adam(params),
        "AdamW": optim.AdamW(params),
        "Adamax": optim.Adamax(params),
        "ASGD": optim.ASGD(params),
        "NAdam": optim.NAdam(params),
        "RAdam": optim.RAdam(params),
        "RMSprop": optim.RMSprop(params),
        "Rprop": optim.Rprop(params),
        "SGD": optim.SGD(params)
    }

def tensorise(obj):
    return torch.as_tensor(obj, dtype=torch.float32, device='cpu')