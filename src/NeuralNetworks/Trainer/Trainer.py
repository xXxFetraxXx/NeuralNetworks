# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ._Trainer_tools import device
from ._Trainer_tools import init_Trainer, init_train
from ._Trainer_tools import epoch_logic, update_lr, update_trakers
from ._Trainer_tools import torch, trange

from .._shared import Module

class Trainer:
    """
    Objet de gestion d'entrainement de modèle.
    """
    def __init__(self                    ,
        *nets            : Module        , # Modèles à entraîner
        inputs           : torch.Tensor  , # Données d'entrée
        outputs          : torch.Tensor  , # Données de sortie
        init_train_size  : float = 0.01  , # Fraction du dataset initiale
        final_train_size : float = 1.0   , # Fraction du dataset finale
        optim            : str   = 'Adam', # Nom de l'optimiseur
        init_lr          : float = 1e-3  , # Learning rate initial
        final_lr         : float = 1e-5  , # Learning rate final
        crit             : str   = 'MSE' , # Fonction de coût
        batch_size       : int   = 1024 ): # Taille des batchs
        """
        Initialise l'entraîneur.
        """
        self.nets   , self.batch_size  = nets, batch_size
        self.init_train_size, self.final_train_size = init_train_size, final_train_size
        self.init_lr, self.final_lr = init_lr, final_lr
        self.inputs , self.outputs  = inputs , outputs
        self.crit   , self.optim_list, self.name       = init_Trainer (
                                            nets       = nets      , # Lie les modèles
                                            crit       = crit      , # Critères
                                            optim      = optim     , # Optimiseurs         
                                            init_lr    = init_lr   , # Ajoute au nom
                                            batch_size = batch_size  # Ajoute au nom
        )
    def train (self                 ,
        num_epochs   : int  = 1500  , # Nombre d'époques
        disable_tqdm : bool = False , # Désactive la barre de progression
        benchmark    : bool = False): # Mode benchmark
        """
        Lance l'entraînement des modèles.
        """
        outputs_size         = self.outputs.size ( ) [1]
        self.inputs, self.outputs, train_losses, train_lrs, n_samples = init_train (
            inputs           = self.inputs          , # Envoi sur le device
            outputs          = self.outputs         , # Envoi sur le device
            init_train_size  = self.init_train_size , # Fraction du dataset initiale
            final_train_size = self.final_train_size, # Fraction du dataset finale
            num_epochs       = num_epochs           , # Utile pour learning rates
            benchmark        = benchmark              # Active le mode benchmark
        )
        for k, net in enumerate (self.nets):
            net = net.to (device)                 # Envoi du réseau sur le device
            net.train ()
            for epoch in trange (num_epochs     , # Nombre d'époques à effectuer
                desc    = f"Training {net.name}", # Paramètre d'affichage
                unit    = "epoch"               , # Paramètre d'affichage
                disable = disable_tqdm         ): # Paramètre d'affichage
                epoch_logic (
                    net          = net                , # Réseau courant
                    epoch        = epoch              , # Epoque actuelle
                    train_losses = train_losses       , # Résidus de l'entrainement
                    n_samples    = n_samples [epoch]  , # Taille des mini-batchs
                    inputs       = self.inputs        , # Données d'entrée
                    outputs      = self.outputs       , # Données de sortie
                    outputs_size = outputs_size       , # Nombre de sorties
                    batch_size   = self.batch_size    , # Taille du batch
                    optim        = self.optim_list [k], # Calcul des gradients
                    crit         = self.crit            # Calcul des résidus
                )
                update_lr (
                    init_lr      = self.init_lr       , # Learning rate initial
                    final_lr     = self.final_lr      , # Learning rate final
                    optim        = self.optim_list [k], # Met à jour le learning rate
                    outputs_size = outputs_size       , # Nombre de sorties
                    train_losses = train_losses       , # Résidus de l'entrainement
                    train_lrs    = train_lrs          , # lrs de l'entrainement
                    epoch        = epoch                # Epoque actuelle
                )
            net = net.to   (torch.device ("cpu")) # Envoi du réseau sur le cpu
            update_trakers (
                net          = net         ,      # Réseau courant
                train_losses = train_losses,      # Met à jour la liste de résidus
                train_lrs    = train_lrs   )      # Met à jour la liste de lr
            net.eval ()
        self.inputs  = self.inputs.to  (torch.device ("cpu")) # Envoi sur le cpu
        self.outputs = self.outputs.to (torch.device ("cpu")) # Envoi sur le cpu
        torch.cuda.empty_cache ( )                            # Vide le cache
    def __repr__ (self):
        return self.name