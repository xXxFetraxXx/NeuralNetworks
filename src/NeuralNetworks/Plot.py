# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .Dependances import *

def compare(img_array, inputs, *nets):
    """
    Affiche, pour chaque réseau, l’erreur absolue entre l’image originale
    et l’image reconstruite par le réseau.

    Chaque réseau doit posséder :
    - une méthode `encoding(x)` (si RFF activé),
    - un module `model` retournant un tenseur de shape (N, 3),
    - une reconstruction compatible avec (H, W, 3).

    Parameters
    ----------
    img_array : np.ndarray of shape (H, W, 3)
        Image originale servant de référence.
    inputs : tensor-like of shape (H*W, 2)
        Coordonnées normalisées des pixels correspondant à chaque point de l'image.
    nets : MLP
        Un ou plusieurs réseaux possédant les méthodes `.encoding()` et `.model()`,
        et l’attribut `.losses`.

    Notes
    -----
    - L’affichage montre la différence absolue entre l’image originale et la prédiction du réseau.
    - Les pertes cumulées sont également tracées pour chaque réseau.
    - Utilise matplotlib en mode interactif.
    """

    # --- Conversion des inputs en tensor et récupération du nombre d'échantillons ---
    inputs, n_samples = tensorise(inputs), inputs.size(0)
    h, w = img_array.shape[:2]

    # --- Configuration de la grille de figure ---
    grid_length = 2 if len(nets) == 1 else len(nets)
    fig = plt.figure(figsize=(5*grid_length, 10))
    gs = GridSpec(2, grid_length, figure=fig)

    # --- Affichage de l'image originale ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.axis('off')
    ax_orig.set_title("Original Image")
    ax_orig.imshow(img_array)

    # --- Préparation du subplot pour les courbes de pertes ---
    ax_loss = fig.add_subplot(gs[0, 1])
    all_losses = [[loss.item() for loss in net.losses] for net in nets]
    ax_loss.set_xlim(1, max(len(lst) for lst in all_losses))
    ax_loss.set_ylim(0, max(max(lst) for lst in all_losses))

    # --- Boucle sur chaque réseau pour afficher l'erreur et les pertes ---
    for k, net in enumerate(nets):
        # Subplot pour l'erreur absolue
        ax = fig.add_subplot(gs[1, k])
        ax.axis('off')
        ax.set_title(net.name)

        # Prédiction et reconstruction de l'image
        pred = net.model(net.encoding(inputs))
        pred_img = pred.reshape(h, w, 3).cpu().detach().numpy()

        # Tracé des pertes cumulées
        ax_loss.plot(np.arange(1, len(all_losses[k])+1), all_losses[k])

        # Affichage de l'erreur absolue
        ax.imshow(np.abs(img_array - pred_img))

    # --- Affichage final ---
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()
    plt.show()
compare.help = fPrintDoc(compare)

def plot(img_array, inputs, *nets):
    """
    Affiche, pour chaque réseau, l’image reconstruite à partir de ses prédictions.

    Parameters
    ----------
    img_array : np.ndarray of shape (H, W, 3)
        Image originale, utilisée pour connaître les dimensions de reconstruction.
    inputs : tensor-like of shape (H*W, 2)
        Coordonnées normalisées des pixels correspondant à chaque point de l'image.
    nets : MLP
        Un ou plusieurs réseaux possédant les méthodes `.encoding()` et `.model()`,
        et l’attribut `.losses`.

    Notes
    -----
    - Cette fonction affiche la prédiction brute.
    - Les pertes cumulées sont également tracées pour chaque réseau.
    - Utilise matplotlib en mode interactif.
    """

    # --- Conversion des inputs en tensor et récupération du nombre d'échantillons ---
    inputs, n_samples = tensorise(inputs), inputs.size(0)
    h, w = img_array.shape[:2]

    # --- Configuration de la grille de figure ---
    grid_length = 2 if len(nets) == 1 else len(nets)
    fig = plt.figure(figsize=(5*grid_length, 10))
    gs = GridSpec(2, grid_length, figure=fig)

    # --- Affichage de l'image originale ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.axis('off')
    ax_orig.set_title("Original Image")
    ax_orig.imshow(img_array)

    # --- Préparation du subplot pour les courbes de pertes ---
    ax_loss = fig.add_subplot(gs[0, 1])
    all_losses = [[loss.item() for loss in net.losses] for net in nets]
    ax_loss.set_xlim(1, max(len(lst) for lst in all_losses))
    ax_loss.set_ylim(0, max(max(lst) for lst in all_losses))

    # --- Boucle sur chaque réseau pour afficher les prédictions et pertes ---
    for k, net in enumerate(nets):
        # Subplot pour l'image reconstruite
        ax = fig.add_subplot(gs[1, k])
        ax.axis('off')
        ax.set_title(net.name)

        # Prédiction et reconstruction de l'image
        pred = net.model(net.encoding(inputs))
        pred_img = pred.reshape(h, w, 3).cpu().detach().numpy()

        # Tracé des pertes cumulées
        ax_loss.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)
        ax_loss.legend()

        # Affichage de l'image prédite
        ax.imshow(pred_img)

    # --- Affichage final ---
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()
    plt.show()
plot.help = fPrintDoc(plot)


def losses(*nets):
    """
    Affiche les courbes de pertes (training loss) de plusieurs réseaux MLP.

    Parameters
    ----------
    nets : MLP
        Un ou plusieurs réseaux possédant un attribut `.losses`
        contenant l'historique des pertes (liste de float).

    Notes
    -----
    - L’axe X correspond aux itérations (epochs ou steps).
    - L’axe Y correspond à la valeur de la perte.
    - La fonction utilise matplotlib en mode interactif pour affichage dynamique.
    """

    # --- Initialisation de la figure ---
    fig = plt.figure(figsize=(5, 5))

    # --- Définition des limites des axes ---
    all_losses = [ [loss.item() for loss in net.losses] for net in nets ]
    plt.xlim(1, max(len(lst) for lst in all_losses)) # X : epochs
    plt.ylim(0, max(max(lst) for lst in all_losses)) # Y : valeurs de pertes

    # --- Tracé des courbes de pertes pour chaque réseau ---
    for k, net in enumerate(nets):
        steps = np.linspace(1, len(net.losses), len(net.losses))  # epochs
        plt.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)

    # --- Affichage ---
    plt.legend()
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()  # mode interactif
    plt.show()
losses.help = fPrintDoc(losses)

def train(inputs, outputs, num_epochs=1500, batch_size=1024, *nets, img_array=None):
    """
    Entraîne un ou plusieurs MLP sur des paires (inputs, outputs) avec gestion optionnelle de l'affichage interactif.

    Affiche dynamiquement si img_array est fourni :
    - L'image originale (référence)
    - Les prédictions des MLP
    - L'évolution des pertes au fil des époques

    Parameters
    ----------
    inputs : array-like
        Entrées du ou des MLP (shape: [n_samples, n_features]).
    outputs : array-like
        Sorties cibles correspondantes (shape: [n_samples, output_dim]).
    num_epochs : int, optional
        Nombre d’époques pour l’entraînement (default=1500).
    batch_size : int, optional
        Taille des mini-batchs pour la descente de gradient (default=1024).
    *nets : MLP
        Un ou plusieurs objets MLP à entraîner.
    img_array : np.ndarray of shape (H, W, 3), optional
        Image de référence pour visualisation des prédictions (default=None).

    Notes
    -----
    - Les MLP sont entraînés indépendamment mais avec le même ordre aléatoire d'échantillons.
    - Utilise torch.amp.GradScaler pour l'entraînement en FP16.
    - La visualisation interactive utilise clear_output() et plt.ion().
    """

    # --- Conversion des données en tensors et récupération du nombre d'échantillons ---
    inputs, outputs, n_samples = tensorise(inputs).to(device), tensorise(outputs).to(device), inputs.size(0)
    for net in nets:
        net.model = net.model.to(device)
    dev = str(device)
    scaler = GradScaler(dev)
    visual = False

    # --- Initialisation de l'affichage interactif si une image de référence est fournie ---
    if img_array is not None:
        visual = True
        h, w = img_array.shape[:2]
        grid_length = 2 if len(nets) == 1 else len(nets)
        fig = plt.figure(figsize=(5*grid_length, 10))
        gs = GridSpec(2, grid_length, figure=fig)

        # Image originale
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.axis('off')
        ax_orig.set_title("Original Image")
        im_orig = ax_orig.imshow(img_array)

        # Images des prédictions initiales des MLP
        ims_preds, axs_preds = [], []
        for k, net in enumerate(nets):
            ax = fig.add_subplot(gs[1, k])
            ax.axis('off')
            ax.set_title(f"Net {k+1}")
            im = ax.imshow(net.model(net.encoding(inputs)).reshape(h, w, 3).cpu().detach().numpy())
            ims_preds.append(im)
            axs_preds.append(ax)

        # Graphiques des pertes
        ax_loss = fig.add_subplot(gs[0, 1])
        lines = []
        for k in range(len(nets)):
            line, = ax_loss.plot([], [], label=f"Network {k+1}")
            lines.append(line)
        ax_loss.set_xlim(1)
        ax_loss.set_ylim(0, 1)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss per Epoch")
        ax_loss.legend()
        ax_loss.set_box_aspect(1)

        fig.canvas.draw_idle()
        plt.tight_layout()
        plt.ion()
        plt.show()

    # --- Boucle principale d’entraînement ---
    for epoch in tqdm(range(num_epochs), desc="train iter"):
        perm = torch.randperm(n_samples, device=device)  # permutation aléatoire des indices
        epoch_loss = [0.0 for _ in nets]  # stockage des pertes par MLP pour l'époque

        # --- Mini-batchs ---
        for k, net in enumerate(nets):
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]

                def closure():
                    """Calcul de la loss et backpropagation pour un mini-batch."""
                    net.optim.zero_grad(set_to_none=True)
                    with autocast(dev):
                        loss = net.crit(net.model(net.encoding(inputs[idx])), outputs[idx])
                    scaler.scale(loss).backward()
                    return loss

                loss = closure()
                scaler.step(net.optim)
                scaler.update()
                epoch_loss[k] += loss

            # --- Stockage des pertes de l'époque ---
            net.losses.append(epoch_loss[k])

        # --- Mise à jour visuelle si mode interactif ---
        if visual:
            with torch.no_grad():
                # Mise à jour des images prédictions
                [im.set_data(nets[k](inputs).reshape(h, w, 3)) for k, im in enumerate(ims_preds)]
                # Mise à jour des courbes de pertes
                all_losses = [[loss.item() for loss in net.losses] for net in nets]
                [line.set_data(np.arange(1, len(all_losses[k])+1), all_losses[k]) for k, line in enumerate(lines)]
                ax_loss.set_xlim(1, max(len(lst) for lst in all_losses))
                ax_loss.set_ylim(0, max(max(lst) for lst in all_losses))
                clear_output(wait=True)
                display(fig)

    # --- Fin du mode interactif ---
    if visual:
        plt.ioff()
train.help = fPrintDoc(train)