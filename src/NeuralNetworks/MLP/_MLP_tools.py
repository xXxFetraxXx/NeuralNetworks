# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import torch, nn, np, pi2, norms, device

def create_layers(
    input_size         : int , #
    output_size        : int , #
    hidden_layers      : int , #
    sigmas             : list, #
    fourier_input_size : int , #
    nb_fourier         : int , #
    norm               : str): #

    if fourier_input_size > input_size:
        raise Exception ("fourier_input_size > input_size impossible")
    if sigmas is None or isinstance(sigmas, int) or (
    isinstance(sigmas, np.ndarray) and sigmas.ndim == 0):
        layer_list = [
            nn.Linear (input_size, hidden_layers [0]),
            norms.get (norm)
        ]
    else:
        layer_list = [
            nn.Linear (2*nb_fourier + input_size-fourier_input_size, hidden_layers [0]),
            norms.get (norm)
        ]

    for k in range (len (hidden_layers) - 1):
        layer_list.extend ([
            nn.Linear (hidden_layers [k], hidden_layers [k + 1]),
            norms.get (norm)
        ])
    layer_list.append (nn.Linear (hidden_layers [-1], output_size))

    return nn.Sequential (*layer_list)

class FourierEncoding (nn.Module):

    def __init__ (self,
        nb_fourier         : int   , # 
        fourier_input_size : int   , #
        sigma              : float): #
        super ().__init__ ()

        self.B = nn.Parameter (torch.randn (nb_fourier, fourier_input_size) * sigma)
        self.size = fourier_input_size

    def forward (self, x : torch.Tensor):
        x_fourier, x_rest = x.split([self.size, x.shape[-1] - self.size], dim=-1)
        vp = pi2 * x_fourier @ self.B.T
        return torch.cat ((torch.cos(vp), torch.sin(vp), x_rest), dim = -1)

def encode (
    input_size         : int , #
    output_size        : int , #
    sigmas             : list, #
    fourier_input_size : int , #
    nb_fourier         : int): #

    if sigmas is None or isinstance(sigmas, int) or (
    isinstance(sigmas, np.ndarray) and sigmas.ndim == 0):
        return nn.ModuleList ([nn.Identity ()]), nn.Identity ()

    size = fourier_input_size

    return (
        nn.ModuleList ([FourierEncoding (nb_fourier, size, sigma) for sigma in sigmas]),
        nn.Linear (len (sigmas) * output_size, output_size)
    )