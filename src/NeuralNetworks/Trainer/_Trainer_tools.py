# NeuralNetworks- Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import torch, trange, scaler, autocast
from .._Dependances import device, dev, crits, optims
from .._shared import Module

def epoch_logic (
    net          : Module                , # Réseau courant
    epoch        : int                   , # Epoche actuelle
    train_losses : torch.Tensor          , # Résidus de l'entrainement
    n_samples    : int                   , # Nombre de donnés à prendre
    inputs       : torch.Tensor          , # Données d'entrée
    outputs      : torch.Tensor          , # Données de sortie
    outputs_size : int                   , # Nombre de sorties
    batch_size   : int                   , # Taille des batchs
    optim        : torch.optim           , # Optimiseur utilisé
    crit         : torch.nn.modules.loss): # Critère de loss utilisé
    """
    Effectue une époque d'entraînement sur des mini-batchs.
    """

    perm = torch.randperm (n_samples, device=device, requires_grad=False)
    
    for i in range (0, n_samples, batch_size):
        idx = perm [i : i + batch_size]
        optim.zero_grad (set_to_none = True)
        
        with autocast (dev):
            all_loss = crit( net.train_forward(inputs[idx]), outputs[idx] ).mean(dim=0)

        scaler.scale  ( all_loss.mean() ).backward ()
        scaler.step   (optim)
        scaler.update (     )
        
        train_losses[epoch].add_(all_loss.detach())

def generate_learning_rate (Nb_iter : int ):
    """
    Génère une courbe de learning rate lisse.
    """

    infl = int (0.1 * Nb_iter)

    lr_curve = torch.empty(Nb_iter, device=device)

    t = torch.linspace(0.0, 1.0, infl, device=device, requires_grad=False)
    t4 = t*t; t3 = t4*t; t4.mul_(t4) ; t5 = t4*t
    lr_curve[:infl] = 1 - 0.5 * (6*t5 - 15*t4 + 10*t3)

    t = torch.linspace(0.0, 1.0, Nb_iter - infl, device=device, requires_grad=False)
    t4 = t*t; t3 = t4*t; t4.mul_(t4) ; t5 = t4*t
    lr_curve[infl:]  =    0.5 * (1 - 6*t5 - 15*t4 + 10*t3)

    return lr_curve

def update_lr (
    init_lr      : float       , # Learning rate initial
    final_lr     : float       , # Learning rate final
    optim        : torch.optim , # Optimiseur utilisé
    outputs_size : int         , # Nombre de sorties
    train_losses : torch.Tensor, # Derniers résidus
    train_lrs    : torch.Tensor, # Learning rates
    epoch        : int        ): # Epoque courante
    """
    Calcule un learning rate adaptatif basé sur les pertes récentes.
    """
    
    if epoch >= 1:
        x = train_losses[max(0, epoch-10):epoch].min(dim=0).values.max()
    else:
        x = train_losses[:1].min(dim=0).values.max()

    y, u9 = x.clone(), x.clone() 

    y.mul_(y); y.mul_(y) 
    u9.mul_(-2); u9.add_(1); u9.addcmul_(x, x, value=1); u9.mul_(u9); u9.mul_(u9)
    
    y.sub_(u9); y.add_(1.0); y.mul_(0.5)

    train_lrs[epoch].clamp_min_(y)
    train_lrs[epoch].mul_(init_lr - final_lr).add_(final_lr)

    for param_group in optim.param_groups:
        param_group ['lr'] = train_lrs[epoch].item()

def update_trakers (
    net          : Module       , # Réseau courant
    train_losses : torch.Tensor , # Résidus
    train_lrs    : torch.Tensor): # Learning rates
    """
    Met à jour l'historique des pertes et le learning rate du modèle.
    """

    net.losses    += train_losses.cpu().tolist()
    net.learnings += train_lrs.cpu().tolist()

        
def init_train (
    inputs           : torch.Tensor, # Données d'entrée
    outputs          : torch.Tensor, # Données de sortie
    init_train_size  : float       , # Proportion de données initiale
    final_train_size : float       , # Proportion de données finale
    num_epochs       : int         , # Nombre d'époques
    benchmark        : bool       ): # Activation du mode benchmark
    """
    Prépare les données et l'environnement d'entraînement.
    """

    torch.backends.cudnn.benchmark    = benchmark
    torch.autograd.set_detect_anomaly  (benchmark)
    torch.autograd.profiler.profile    (benchmark)
    torch.use_deterministic_algorithms (benchmark)


    n_samples = torch.linspace(
        inputs.size(0) * init_train_size,
        inputs.size(0) * final_train_size,
        num_epochs, device = device, requires_grad=False
    ).ceil().int()

    inputs  = inputs.to  (device)
    outputs = outputs.to (device)

    train_lrs    = generate_learning_rate (num_epochs)
    train_losses = torch.zeros(
        (num_epochs, outputs.size(1)), device=device, requires_grad=False
    )
    
    torch.cuda.empty_cache ()
    return inputs, outputs,train_losses, train_lrs, n_samples

def init_Trainer (
    nets       : list , # Modèles à entraîner
    crit       : str  , # Fonction de coût
    optim      : str  , # Optimiseur utilisé
    init_lr    : float, # Learning rate initial
    batch_size : int ): # Taille des batchs
    """
    Initialise le critère de perte et les optimiseurs.
    """
    name =  f"| optim      : {optim}\n"
    name += f"| crit       : {crit}\n"
    name += f"| init_lr    : {init_lr}\n"
    name += f"| batch_size : {batch_size}"

    optim_list = []
    for net in nets:
        param = [{"params" : net.parameters (), "lr" : init_lr}]
        optim_list.append( optims.get (optim) (param))
    return crits.get (crit), optim_list, name