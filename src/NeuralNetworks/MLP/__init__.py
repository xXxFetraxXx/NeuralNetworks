# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import norm_list, nn
from .FourierFeatures import encode
from .Layers import create_layers
from .inference import infer

class MLP (nn.Module):
    """
    Multi-Layer Perceptron avec encodage de Fourier optionnel.

    Cette classe implémente un MLP configurable, pouvant intégrer un ou
    plusieurs encodages de Fourier en entrée afin d'améliorer la capacité
    de représentation sur des signaux à haute fréquence.

    Parameters
    ----------
    input_size : int, optional
        Dimension des entrées du réseau. Default: 1.
    output_size : int, optional
        Dimension des sorties du réseau. Default: 1.
    hidden_layers : list[int], optional
        Liste des tailles des couches cachées. Default: [1].
    sigmas : list[float] ou None, optional
        Paramètres sigma pour les encodages de Fourier. Si `None`,
        aucun encodage de Fourier n'est utilisé.
    fourier_input_size : int, optional
        WIP
    nb_fourier : int, optional
        Nombre de composantes de Fourier par encodage. Default: 8.
    norm : str, optional
        Nom de la fonction d'activation à utiliser. Default: "Relu".
    name : str, optional
        Nom du réseau. Default: "Net".
    """

    def __init__ (
        self,
        input_size = 1,
        output_size = 1,
        hidden_layers = [1],
        sigmas = None,
        fourier_input_size = 2,
        nb_fourier = 8,
        norm = "Relu",
        name = "Net"):
        super ().__init__ ()

        # --- Activation ---
        self.norm = norm_list.get (norm)
        if self.norm is None:
            print (f"Warning: '{norm}' not recognized, falling back to 'm is'")
            self.norm = norm_list.get ("Relu")

        # --- Attributs ---
        self.losses = []
        self.learnings = []
        self.name = name
        
        ## --- Encodage Fourier ou passthrough ---
        self.encodings, self.f = encode (
            input_size,
            output_size,
            sigmas,
            fourier_input_size,
            nb_fourier
        )
            
        # --- Construction du réseau ---
        self.model = create_layers (
            input_size,
            output_size,
            hidden_layers,
            sigmas,
            fourier_input_size,
            nb_fourier,
            self.norm
        )

    def forward (self, x):
        """
        Effectue une passe avant du réseau.

        Parameters
        ----------
        x : torch.Tensor
            Entrées du réseau de shape `(N, input_size)`.

        Returns
        -------
        torch.Tensor
            Sortie du MLP de shape `(N, output_size)`.
        """
        return infer (self, x)