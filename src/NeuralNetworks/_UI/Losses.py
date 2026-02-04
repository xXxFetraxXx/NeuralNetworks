# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import plt, np, FixedLocator, torch
from ._plot import plot, init_plot

def losses (*nets                 , #
    fuse_losses : bool = True     , #
    names       : list = None     , #
    fig_size    : int = 5         , #
    color       : str = "#888888"): #

    fig, ax = init_plot (fig_size, color)

    if fuse_losses:
        all_losses = [[np.mean (losses) for losses in net.losses] for net in nets]
    else:
        all_losses = [net.losses for net in nets]
    
    if max (len (lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max (len (lst) for lst in all_losses)

    all_losses = np.array (all_losses)
    plt.xlim (1, lenlosses)
    plt.ylim (10 ** (np.floor (np.log10 (np.min (all_losses)))),
              10 ** (np.ceil  (np.log10 (np.max (all_losses)))))

    if fuse_losses:

        for k, net in enumerate (nets):
            ax.plot (np.arange(1, len (all_losses [k]) + 1),
                all_losses[k], label=net.name)
    else:

        if names is None:
            names = range (all_losses.shape [-1])
        for k, net in enumerate(nets):
            for i in range (all_losses.shape [-1]):
                ax.plot (np.arange (1, len (all_losses [k] [:, i]) + 1),
                    all_losses [k] [:, i], label = f"{net.name} : {names [i]}")

    plot (ax, "Epochs", "Résidus", "")