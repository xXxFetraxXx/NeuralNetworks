# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import norm_list, nn, torch, device

class Module (nn.Module):

    def __init__ (self, name = "Net"):
        super ().__init__ ()

        # --- Attributs ---
        self.losses = []
        self.learnings = []
        self.name = name

    def _forward (self,x):
        raise Exception ("_forward n'est pas défini dans la classe")

    def train_forward (self,x):
        raise Exception ("train_forward n'est pas défini dans la classe")
            
    def forward (self, x):

        with torch.no_grad ():
            x = x.unsqueeze (0) if x.dim () == 1 else x
            self = self.to (device)
            x = x.to (device)

            output = self._forward(x).cpu ().numpy ().flatten ()
            
            x = x.to ('cpu')
            self = self.to ('cpu')

            return output

            
    