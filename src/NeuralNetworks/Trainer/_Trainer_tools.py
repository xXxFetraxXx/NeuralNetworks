# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import torch, scaler, autocast, trange
from .._Dependances import device, dev, crits, optims, CompositeCriterion
from .._shared import Module


def lhs_1d(n_samples: int, population_size: int) -> torch.Tensor:
    """LHS 1D : 2 allocations, 6 kernels.

    Le clamp max est indispensable : en float32, (i + u) * scale peut
    arrondir a population_size (ex: 183749.999 -> 183750.0), produisant
    un indice hors limites.
    """
    r = torch.rand(n_samples, device=device)     # kernel 1 : u ~ U[0,1)
    r.add_(torch.arange(n_samples, device=device, dtype=torch.float32))  # kernel 2 (arange) + 3 (add_)
    r.mul_(population_size / n_samples)                                # kernel 4
    r.clamp_(max=population_size - 1)                                  # kernel 5
    return r.to(torch.int64)                                           # kernel 6

def resolve_crit(crit):
    """Transforme str | list en un seul critère appelable (net, inputs, outputs).

    - str               → crits.get(name)
    - list[str]         → CompositeCriterion, poids = 1.0
    - list[tuple]       → CompositeCriterion, (name, weight)
    """
    if isinstance(crit, str):
        return crits.get(crit)
    if isinstance(crit, list):
        components = []
        for item in crit:
            if isinstance(item, str):
                components.append((crits.get(item), 1.0))
            else:
                name, weight = item
                components.append((crits.get(name), weight))
        return CompositeCriterion(components)
    return crit


def epoch_logic(
    net: Module,
    epoch: int,
    train_losses: torch.Tensor,
    n_samples: int,
    inputs: torch.Tensor,
    inputs_size: int,
    outputs: torch.Tensor | None,
    outputs_size: int,
    batch_size: int,
    optim: torch.optim,
    crit,
):
    """Effectue une époque d'entraînement sur des mini-batchs."""
    sample_idx = lhs_1d(int(n_samples.item()), inputs_size)

    for i in range(0, n_samples, batch_size):
        idx = sample_idx[i:i + batch_size]
        optim.zero_grad(set_to_none=True)

        with autocast(dev):
            all_loss = crit(net, inputs[idx], outputs[idx])

        scaler.scale(all_loss.mean()).backward()
        scaler.step(optim)
        scaler.update()

        train_losses[epoch].add_(all_loss.detach())


def generate_learning_rate(nb_iter: int):
    """Génère une courbe de learning rate lisse."""
    infl = int(0.1 * nb_iter)

    lr_curve = torch.empty(nb_iter, device=device)

    t = torch.linspace(0.0, 1.0, infl, device=device, requires_grad=False)
    t4 = t * t
    t3 = t4 * t
    t4.mul_(t4)
    t5 = t4 * t
    lr_curve[:infl] = 1 - 0.5 * (6 * t5 - 15 * t4 + 10 * t3)

    t = torch.linspace(
        0.0, 1.0, nb_iter - infl, device=device, requires_grad=False
    )
    t4 = t * t
    t3 = t4 * t
    t4.mul_(t4)
    t5 = t4 * t
    lr_curve[infl:] = 0.5 * (1 - 6 * t5 + 15 * t4 - 10 * t3)

    return lr_curve.contiguous()


def update_lr(
    init_lr: float,
    final_lr: float,
    optim: torch.optim ,
    outputs_size: int,
    train_losses: torch.Tensor,
    train_lrs: torch.Tensor,
    epoch: int,
):
    """Calcule un learning rate adaptatif basé sur les pertes récentes."""
    if epoch >= 1:
        x = train_losses[max(0, epoch - 10):epoch].min(dim=0).values.max()
    else:
        x = train_losses[:1].min(dim=0).values.max()

    y = x.clone()
    u9 = x.clone()

    y.mul_(y)
    y.mul_(y)
    u9.mul_(-2)
    u9.add_(1)
    u9.addcmul_(x, x, value=1)
    u9.mul_(u9)
    u9.mul_(u9)

    y.sub_(u9)
    y.add_(1.0)
    y.mul_(0.5)

    train_lrs[epoch].clamp_min_(y)
    train_lrs[epoch].mul_(init_lr - final_lr).add_(final_lr)

    for param_group in optim.param_groups:
        param_group['lr'] = train_lrs[epoch].item()


def update_trakers(
    net: Module,
    train_losses: torch.Tensor,
    train_lrs: torch.Tensor,
):
    """Met à jour l'historique des pertes et le learning rate du modèle."""
    net.losses += train_losses.cpu().tolist()
    net.learnings += train_lrs.cpu().tolist()


def init_train(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    init_train_size: float,
    final_train_size: float,
    num_epochs: int,
    benchmark: bool,
):
    """Prépare les données et l'environnement d'entraînement."""
    torch.backends.cudnn.benchmark = benchmark
    torch.autograd.set_detect_anomaly(benchmark)
    torch.autograd.profiler.profile(benchmark)
    torch.use_deterministic_algorithms(benchmark)

    n_samples = torch.linspace(
        inputs.size(0) * init_train_size,
        inputs.size(0) * final_train_size,
        num_epochs, device=device, requires_grad=False,
    ).ceil().int()

    """
    
    if device.type == 'cuda':
        inputs = inputs.pin_memory()
        outputs = outputs.pin_memory()
    """
    inputs = inputs.to(device, non_blocking=True).contiguous()
    outputs = outputs.to(device, non_blocking=True).contiguous()

    train_lrs = generate_learning_rate(num_epochs).contiguous()
    train_losses = torch.zeros(
        (num_epochs, outputs.size(1)), device=device, requires_grad=False,
    ).contiguous()

    torch.cuda.empty_cache()
    return inputs, outputs, train_losses, train_lrs, n_samples


def init_Trainer(
    nets: list,
    crit,
    optim: str,
    init_lr: float,
    batch_size: int,
):
    """Initialise le critère de perte et les optimiseurs."""
    name = f"| optim      : {optim}\n"
    name += f"| crit       : {crit}\n"
    name += f"| init_lr    : {init_lr}\n"
    name += f"| batch_size : {batch_size}"

    optim_list = []
    for net in nets:
        param = [{"params": net.parameters(), "lr": init_lr}]
        optim_list.append(optims.get(optim)(param))
    return resolve_crit(crit), optim_list, name
