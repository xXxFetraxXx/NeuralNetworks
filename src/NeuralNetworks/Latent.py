# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .Dependances import *

class Latent(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        
        self.insize = insize
        self.outsize = outsize
        
        # Start latent conv channels
        channels = 128
        
        # ----- Encoder -----
        layers = []
        for k in range(5):
            layers.append(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1))
            channels = channels // 2
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels * 1 * 1, outsize))  # adjust if spatial dims not 1x1

        self.Emodel = nn.Sequential(*layers).to(device)

        # ----- Decoder -----
        layers = []
        layers.append(nn.Linear(outsize, channels))  # output same number of channels
        layers.append(nn.Unflatten(1, (channels, 1, 1)))

        for k in range(5):
            layers.append(nn.ConvTranspose2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1))
            channels = channels * 2
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        self.Dmodel = nn.Sequential(*layers).to(device)

    def encode(self, image):
        return self.Emodel(image)
        
    def decode(self, vector):
        return self.Dmodel(vector)
