# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import torch, nn

def create_layers(
    input_size,
    output_size,
    hidden_layers,
    sigmas,
    fourier_input_size,
    nb_fourier,
    norm):

    layer_list = [
        nn.Linear (input_size if sigmas is None else 2 * nb_fourier, hidden_layers [0]),
        norm
    ]

    for k in range (len (hidden_layers) - 1):
        layer_list.extend ([
            nn.Linear (hidden_layers [k], hidden_layers [k + 1]),
            norm
        ])
    layer_list.append (nn.Linear (hidden_layers [-1], output_size))

    return torch.jit.script (nn.Sequential (*layer_list))