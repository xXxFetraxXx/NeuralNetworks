# NeuralNetworks - Convolutional Neural Networks
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ._CNN_tools import create_layers, torch
from .._shared import Module


class CNN(Module):
    """
    Réseau de neurones convolutif CNN.
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_size: int = 1,
        conv_layers: list = [1],
        kernel_size: int = 3,
        norm: str = "Relu",
        name: str = "Net",
    ):
        """Donnés de reconstruction de l'objet."""
        super().__init__(
            name,
            in_channels=in_channels,
            output_size=output_size,
            conv_layers=conv_layers,
            kernel_size=kernel_size,
            norm=norm,
            name=name,
        )

        self.model = create_layers(
            in_channels=in_channels,
            output_size=output_size,
            conv_layers=conv_layers,
            kernel_size=kernel_size,
            norm=norm,
        )

    def _forward(self, x: torch.Tensor):
        """Forward pass interne."""
        return self.model(x)

    @property
    def _dummy_input(self):
        """Données d'entrées pour enregistrement en .onnx."""
        return torch.randn(1, self.Reconstruction_data["in_channels"], 32, 32)
