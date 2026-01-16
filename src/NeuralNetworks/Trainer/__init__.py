# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import crit_list, optim_list
from .train import train_f
from .sample_data import sample_data

class Trainer:

    def __init__(self,
        *nets,
        inputs,
        outputs,
        test_size = None,
        optim = 'Adam',
        init_lr = 0.01,
        crit = 'MSE',
        batch_size = float):

        self.batch_size = batch_size
        self.nets = nets
        self.init_lr = init_lr

        self.X_train, self.X_test, self.y_train, self.y_test = sample_data (
            inputs,
            outputs,
            test_size
        )

        # --- Fonction de perte ---
        self.crit = crit_list.get(crit)
        if self.crit is None:
            print(f"Warning: '{self.crit}' not recognized, falling back to 'MSE'")
            self.crit = crit_list.get("MSE")

        # --- Sélection de l’optimiseur ---
        self.optims = []
        for net in nets:
            params = [{"params": net.parameters(), "lr": self.init_lr}]
            new_optim = optim_list(params).get(optim)
            if new_optim is None:
                print(f"Warning: '{optim}' not recognized, falling back to 'Adam'")
                new_optim = optim_list(params).get("Adam")
            self.optims.append(new_optim)

    def train (self, num_epochs = 1500, activate_tqdm = True):
        train_f (self, num_epochs, activate_tqdm)