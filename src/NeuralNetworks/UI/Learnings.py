# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import plt, np

def learnings(*nets):

    # --- Initialisation de la figure ---
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)

    # --- Définition des limites des axes ---
    all_learnings = [[lr for lr in net.learnings] for net in nets]
    if max(len(lst) for lst in all_learnings) == 1:
        lenlearnings = 2
    else:
        lenlearnings = max(len(lst) for lst in all_learnings)
    plt.xlim(1, lenlearnings)

    # --- Tracé des courbes de pertes pour chaque réseau ---
    for k, net in enumerate(nets):
        ax1.plot(
            np.arange(1, len(all_learnings[k]) + 1),
            all_learnings[k],
            label=net.name
        )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Learning rate")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    
    plt.yscale('log', nonpositive='mask')
    # --- Affichage ---
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()  # mode interactif
    plt.show()