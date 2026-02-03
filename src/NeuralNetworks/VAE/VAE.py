# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import *
from .._shared import Module

class VAE(Module):
    def __init__(self                           , #
        imsize          : int                   , #
        latentsize      : int                   , #
        labelsize       : int                   , #
        channels        : list = [16, 32, 16, 8], #
        linear_channels : list = [100]          , #
        name            : str  = "encoder"      , #
        norm            : str  = "Relu"         , #
        norm_cc         : str  = "Relu"        ): #

        super().__init__(name                , #
            imsize          = imsize         , #
            latentsize      = latentsize     , #
            labelsize       = labelsize      , #
            channels        = channels       , #
            linear_channels = linear_channels, #
            name            = name           , #
            norm            = norm           , #
            norm_cc         = norm_cc        ) #

        self.imsize     = imsize
        self.latentsize = latentsize
        self.labelsize  = labelsize
        
        # Start latent conv channels
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norms.get(norm)

        if norm_cc is None:
            self.norm_cc = nn.Identity()
        else:
            self.norm_cc = norms.get(norm_cc)
        
        # ----- Encoder -----
        Elayers = []
        in_ch = 1  # grayscale input
        for out_ch in channels:
            Elayers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            Elayers.append(self.norm_cc)
            Elayers.append(nn.MaxPool2d(kernel_size=3, stride=3, padding=1))
            in_ch = out_ch

        # Compute final flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, imsize, imsize)
            for layer in Elayers:
                dummy = layer(dummy)
            flat_dim = dummy.numel()
            del dummy
        
        Elayers.append(nn.Flatten())
        Elayers.append(nn.Linear(flat_dim, linear_channels[0]))#(cl_nbr+1) * int(imsize/2**cl_nbr)**2, latentsize * 30))
        Elayers.append(self.norm)
        Elayers.append(nn.Linear(linear_channels[0], latentsize))
        Elayers.append(self.norm)
        self.Emodel = nn.Sequential(*Elayers)
        
        # ----- Decoder -----
        Dlayers = []
        Dlayers.append(nn.Linear(latentsize, int((labelsize+latentsize)/2)))
        Dlayers.append(self.norm)
        Dlayers.append(nn.Linear(int((labelsize+latentsize)/2), int((labelsize+latentsize))))
        Dlayers.append(self.norm)
        Dlayers.append(nn.Linear(int((labelsize+latentsize)), int((labelsize+latentsize))))
        Dlayers.append(self.norm)
        Dlayers.append(nn.Linear(int((labelsize+latentsize)), labelsize))
        Dlayers.append(self.norm)
     
        self.Dmodel = nn.Sequential(*Dlayers)

    def encode(self, inputs):
        image = np.array(inputs)
        inputs = tensorise(inputs).to(device)
        
        if image.ndim == 4:
            x = inputs
        elif inputs.ndim == 3:        # [H, W, C]? Or [C, H, W]?
            x = inputs.unsqueeze(0) # → [1, C, H, W]
        elif image.ndim == 2:      # [H, W]
            x = inputs.unsqueeze(0).unsqueeze(0)  # → [1, 1, H, W]
        inputs = inputs.to('cpu')

        self.Emodel = self.Emodel.to(device)
        output = self.Emodel(x).flatten()
        self.Emodel = self.Emodel.to('cpu')
        return output
                                                
    def decode(self, vector):
        vector = tensorise(vector).to(device)
        x = vector.view(1, 1, 1, self.latentsize)   # batch=1, channels=8, h=1, w=1
        vector = tensorise(vector).to('cpu')
        self.Dmodel = self.Dmodel.to(device)
        output = self.Dmodel(x).cpu().detach().numpy()[0][0]
        self.Dmodel = self.Dmodel.to('cpu')
        return output

    def _forward (self, x):
        return self.Dmodel(x)

    def train_forward (self, x):
        return self.Dmodel(self.Emodel(x))