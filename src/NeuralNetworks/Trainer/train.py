# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import torch, GradScaler, device, tqdm, autocast

from .dynamic_learning_rate import generate_learning_rate, update_lr

def train_f (Trainer, num_epochs = 1, activate_tqdm = True):
        dev = str (device)
        scaler = GradScaler (dev)

        lrs = generate_learning_rate (num_epochs, Trainer.init_lr)

        Trainer.X_train = Trainer.X_train.to (device)
        Trainer.y_train = Trainer.y_train.to (device)
        n_samples = Trainer.X_train.size (0)

        torch.cuda.empty_cache ()
        for k, net in enumerate (Trainer.nets):
            net = net.to (device)
            net.learnings.append(Trainer.init_lr)

            pbar = tqdm (
                range (num_epochs),
                desc = f"train epoch",
                disable = not (activate_tqdm)
            )

            for epoch in pbar:
                # Génération d'un ordre aléatoire des indices
                perm = torch.randperm (n_samples, device = device)
                epoch_loss = 0.0
    
                # --- Parcours des mini-batchs ---
                for i in range (0, n_samples, Trainer.batch_size):
                    idx = perm [i : i + Trainer.batch_size]
    
                    # Fonction interne calculant la perte et les gradients
                    def closure ():
                        Trainer.optims [k].zero_grad (set_to_none = True)
                        with autocast (dev):
                            loss = Trainer.crit (
                                net.f (
                                    torch.cat (
            [net.model (encoding (Trainer.X_train [idx]))for encoding in net.encodings],
            dim = 1
                                    )
                                ),
                                Trainer.y_train[idx]
                            )
                            scaler.scale (loss).backward ()
                            return loss
    
                    epoch_loss += closure()
                    scaler.step (Trainer.optims [k])
                    scaler.update ()
    
                # --- Stockage de la perte de l'époque ---
                #Trainer.frequencies.append(net.encodings[0].B.detach().cpu().clone())
                net.losses.append (epoch_loss.item ())
                net.learnings.append (update_lr (net.losses [-20:], lrs, epoch, net.learnings[-1]))
                for param_group in Trainer.optims [k].param_groups:
                    param_group ['lr'] = net.learnings[-1]
                
                pbar.set_postfix(loss=f"{epoch_loss:.5f}",lr=f"{net.learnings[-1]:.5f}")
                
            net = net.to ('cpu')
            net.learnings.pop(-1)
        Trainer.X_train = Trainer.X_train.to ('cpu')
        Trainer.y_train = Trainer.y_train.to ('cpu')
        torch.cuda.empty_cache ()