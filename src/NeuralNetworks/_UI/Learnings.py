# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 - 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import plt, np, FixedLocator
from ._plot import plot, init_plot

def learnings (*nets           , #
    fig_size : int = 5         , #
    color    : str = "#888888"): #

    fig, ax = init_plot (fig_size, color)
    all_learnings = [[lr for lr in net.learnings] for net in nets]

    if max (len (lst) for lst in all_learnings) == 1:
        lenlearnings = 2
    else:
        lenlearnings = max (len (lst) for lst in all_learnings)

    plt.xlim (1, lenlearnings)
    plt.ylim (10 ** (np.floor (np.log10 (np.min (all_learnings)))),
              10 ** (np.ceil  (np.log10 (np.max (all_learnings)))))
    
    for k, net in enumerate (nets):
        ax.plot (np.arange(1, len (all_learnings [k]) + 1),
            all_learnings [k], label = net.name)

    plot (ax, "Epochs", "Taux d'apprentissage", "")