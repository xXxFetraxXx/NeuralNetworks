# NeuralNetworks - Convolutional Neural Networks
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import nn, torch, norms


def create_layers(
    in_channels: int,
    output_size: int,
    conv_layers: list,
    kernel_size: int,
    norm: str,
):
    layer_list = []
    prev = in_channels

    for out_ch in conv_layers:
        layer_list.extend([
            nn.Conv2d(prev, out_ch, kernel_size, padding=kernel_size // 2),
            norms.get(norm),
            nn.MaxPool2d(2),
        ])
        prev = out_ch

    layer_list.append(nn.AdaptiveAvgPool2d((1, 1)))
    layer_list.append(nn.Flatten())
    layer_list.append(nn.Linear(prev, output_size))

    return nn.Sequential(*layer_list)
