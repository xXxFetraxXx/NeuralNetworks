# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import plt, np

def losses(*nets):

    # --- Initialisation de la figure ---
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)

    # --- Définition des limites des axes ---
    all_losses = [[loss for loss in net.losses] for net in nets]
    if max(len(lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max(len(lst) for lst in all_losses)
    plt.xlim(1, lenlosses)

    # --- Tracé des courbes de pertes pour chaque réseau ---
    for k, net in enumerate(nets):
        ax1.plot(
            np.arange(1, len(all_losses[k]) + 1),
            all_losses[k],
            label=net.name
        )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    
    plt.yscale('log', nonpositive='mask')
    # --- Affichage ---
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Résidus")
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()  # mode interactif
    plt.show()