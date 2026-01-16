# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import *
from torchvision.datasets import MNIST

def data(path):
    """
    Charge le dataset MNIST depuis `path`, applique une transformation en tenseur,
    puis convertit les images en vecteurs numpy aplatis et les labels en tenseur PyTorch.

    Parameters
    ----------
    path : str
        Chemin du dossier où MNIST sera téléchargé ou chargé.

    Returns
    -------
    inputs : np.ndarray
        Tableau numpy de shape (N, 784) contenant les images MNIST aplaties.
        Chaque pixel est normalisé dans [0, 1] via `ToTensor()`.
    outputs : torch.Tensor
        Tenseur PyTorch de shape (N, 1) contenant les labels entiers (0–9).

    Notes
    -----
    - Le dataset MNIST est téléchargé si absent.
    - Chaque image 28×28 est convertie via `ToTensor()` puis aplatie en vecteur de 784 valeurs.
    - Les labels sont convertis en tenseur long et remis dans une dimension (N, 1)
      pour compatibilité avec un MLP produisant une sortie scalaire.
    """
    transform = Compose([ToTensor()])
    dataset = MNIST(path, transform=transform, download=True)
    
    inputs, outputs = [], []
    for data in dataset:
        outputs.append(data[1])
        value= data[0].numpy().flatten()
        inputs.append(value)
    outputs = torch.tensor(np.array(outputs))   # convert list → tensor
    outputs = outputs.unsqueeze(1) 
    inputs = np.array(inputs)

    return inputs, outputs

def evaluate (inputs, *nets):
    """
    Évalue visuellement un ou plusieurs réseaux sur un échantillon MNIST choisi
    aléatoirement. La fonction affiche simultanément :

        - l'image d'entrée (28×28),
        - les courbes de perte de chaque réseau (échelle logarithmique),
        - la prédiction de chaque réseau imprimée dans la console.

    Parameters
    ----------
    inputs : np.ndarray
        Tableau numpy contenant les images aplaties (N, 784).
        Une image sera choisie aléatoirement parmi celles-ci.
    nets : MLP
        Un ou plusieurs réseaux entraînés, chacun possédant :
        - net.losses : liste des pertes par époque,
        - net.name   : nom du modèle,
        - net(x)     : méthode d'inférence retournant une valeur prédite.

    Notes
    -----
    - L'image affichée est l'entrée sélectionnée, remise en forme en 28×28.
    - Les pertes sont tracées pour chaque réseau sur une échelle Y logarithmique.
    - Les prédictions sont arrondies et converties en entiers pour un affichage clair.
    - Une figure matplotlib avec deux sous-graphiques est générée via GridSpec :
        * à gauche  : l'image MNIST,
        * à droite : les courbes de pertes.
    - Les résultats (prédictions) sont également affichés dans la console.
    """

    # --- Configuration de la grille de figure ---
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, figure=fig)

    index = np.random.randint(0,len(inputs)-1)

    # --- Préparation du subplot pour les courbes de pertes ---
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_loss.set_yscale('log', nonpositive='mask')
    all_losses = [[loss for loss in net.losses] for net in nets]
    if max(len(lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max(len(lst) for lst in all_losses)
    ax_loss.set_xlim(1, lenlosses)

    preds = []
    for k, net in enumerate(nets):
        preds.append(int(np.round(net(inputs[index]))))
        # Tracé des pertes cumulées
        ax_loss.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)
    ax_loss.legend()

    # --- Affichage de l'image originale ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.axis('off')
    ax_orig.set_title("input")
    show = inputs[index].reshape(28,28)
    ax_orig.imshow(255*show)

    # --- Affichage final ---
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()
    plt.show()

    for k in rglen(preds):
        print(f"{nets[k].name} output : {preds[k]}")