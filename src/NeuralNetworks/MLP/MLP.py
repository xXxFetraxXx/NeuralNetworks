# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ._MLP_tools import encode, create_layers, torch
from .._shared import Module

class MLP (Module):
    """
    Réseau de neurones MLP avec encodage optionnel de type Fourier.
    """
    def __init__ (self                    ,
        input_size         : int  = 1     , # Dimension d'entrée
        output_size        : int  = 1     , # Dimension de sortie
        hidden_layers      : list = [1]   , # Tailles des couches cachées
        sigmas             : list = None  , # Répartition des fréquences
        fourier_input_size : int  = 1     , # Dimension d'entrée pour l'encodage Fourier
        nb_fourier         : int  = 8     , # Nombre de composantes Fourier
        norm               : str  = "Relu", # Fonction d'activation / normalisation
        name               : str  = "Net"): # Nom du modèle

        """   Donnés de reconstruction de l'objet   """
        super ().__init__ (name                     , #
            input_size         = input_size         , # 
            output_size        = output_size        , # 
            hidden_layers      = hidden_layers      , # 
            sigmas = 0 if sigmas is None else sigmas, # 
            fourier_input_size = fourier_input_size , #       
            nb_fourier         = nb_fourier         , #
            norm               = norm               , #
            name               = name               ) #

        self.encodings, self.f = encode (
            input_size         = input_size        , # Couche d'adaptation d'entrée
            output_size        = output_size       , # Couche d'adaptation de sortie
            sigmas             = sigmas            , # Répartition des fréquences
            fourier_input_size = fourier_input_size, # Encode les premiers inputs
            nb_fourier         = nb_fourier          # Attribue les fréquences
        )
        self.model             = create_layers (
            input_size         = input_size        , # Créé une couche d'entrée
            output_size        = output_size       , # Créé une couche de sortie
            hidden_layers      = hidden_layers     , # Créé des couches intermédiaires
            sigmas             = sigmas            , # Répartition des fréquences
            fourier_input_size = fourier_input_size, # Dimension d'encodage Fourier
            nb_fourier         = nb_fourier        , # Nombre de composantes Fourier
            norm               = norm                # Fonction d'activation
        )
    def _forward (self, x : torch.Tensor):
        """
        Forward pass interne avec concaténation des encodages.
        """
        results_list = [self.model (encoding (x)) for encoding in self.encodings]
        return self.f (torch.cat (results_list, dim = 1))

    @property
    def _dummy_input(self):
        """
        Données d'entrées pour enregistrement en .onnx
        """
        return torch.randn(1, self.Reconstruction_data["input_size"])