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

    def __init__(
        self,
        *nets: Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        init_train_size: float = 0.01,
        final_train_size: float = 1.0,
        optim: str = 'Adam',
        init_lr: float = 1e-3,
        final_lr: float = 1e-5,
        crit: str | list = 'MSE',
        batch_size: int = 1024,
    ):
        """Initialise l'entraîneur."""
        self.nets = nets
        self.batch_size = batch_size
        self.init_train_size = init_train_size
        self.final_train_size = final_train_size
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.inputs = inputs
        self.outputs = outputs
        self.crit, self.optim_list, self.name = init_Trainer(
            nets=nets,
            crit=crit,
            optim=optim,
            init_lr=init_lr,
            batch_size=batch_size,
        )

    def train(
        self,
        num_epochs: int = 1500,
        disable_tqdm: bool = False,
        benchmark: bool = False,
    ):
        """Lance l'entraînement des modèles."""
        outputs = self.outputs if self.outputs is not None else self.inputs
        inputs_size = outputs.size(0)
        outputs_size = outputs.size(1)
        self.inputs, outputs, train_losses, train_lrs, n_samples = (
            init_train(
                inputs=self.inputs,
                outputs=outputs,
                init_train_size=self.init_train_size,
                final_train_size=self.final_train_size,
                num_epochs=num_epochs,
                benchmark=benchmark,
            )
        )
        for k, net in enumerate(self.nets):
            net = net.to(device)
            net.train()
            for epoch in trange(
                num_epochs,
                desc=f"Training {net.name}",
                unit="epoch",
                disable=disable_tqdm,
            ):
                epoch_logic(
                    net=net,
                    epoch=epoch,
                    train_losses=train_losses,
                    n_samples=n_samples[epoch],
                    inputs=self.inputs,
                    inputs_size=inputs_size,
                    outputs=outputs,
                    outputs_size=outputs_size,
                    batch_size=self.batch_size,
                    optim=self.optim_list[k],
                    crit=self.crit,
                )
                update_lr(
                    init_lr=self.init_lr,
                    final_lr=self.final_lr,
                    optim=self.optim_list[k],
                    outputs_size=outputs_size,
                    train_losses=train_losses,
                    train_lrs=train_lrs,
                    epoch=epoch,
                )
            net = net.to(torch.device("cpu"))
            update_trakers(
                net=net,
                train_losses=train_losses,
                train_lrs=train_lrs,
            )
            net.eval()
        self.inputs = self.inputs.to(torch.device("cpu"))
        torch.cuda.empty_cache()

    def __repr__(self):
        return self.name
