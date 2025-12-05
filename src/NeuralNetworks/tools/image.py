# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import *

def url(url, img_size=256):
    """
    Télécharge une image depuis une URL, la redimensionne et prépare les
    données pour l'entraînement d'un MLP pixel-wise.

    Cette fonction retourne :
    - `img_array` : image RGB sous forme de tableau NumPy (H, W, 3), pour affichage.
    - `inputs` : coordonnées normalisées (x, y) de chaque pixel, sous forme de tenseur (H*W, 2).
    - `outputs` : valeurs RGB cibles pour chaque pixel, sous forme de tenseur (H*W, 3).

    Paramètres
    ----------
    url : str
        URL de l'image à télécharger.
    img_size : int, optionnel
        Taille finale carrée de l'image (img_size x img_size). Par défaut 256.

    Retours
    -------
    img_array : numpy.ndarray of shape (H, W, 3)
        Image sous forme de tableau NumPy, valeurs normalisées entre 0 et 1.
    inputs : torch.Tensor of shape (H*W, 2)
        Coordonnées normalisées des pixels pour l'entrée du MLP.
    outputs : torch.Tensor of shape (H*W, 3)
        Valeurs RGB cibles pour chaque pixel, pour la sortie du MLP.

    Notes
    -----
    - La fonction utilise `PIL` pour le traitement de l'image et `torchvision.transforms`
      pour la conversion en tenseur normalisé.
    - Les coordonnées sont normalisées dans [0, 1] pour une utilisation optimale
      avec des MLP utilisant Fourier Features ou activations standard.
    - Les tenseurs `inputs` et `outputs` sont prêts à être envoyés sur GPU si nécessaire.
    """

    # --- Téléchargement et ouverture de l'image ---
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # --- Redimensionnement et conversion en tenseur normalisé ---
    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor()  # Donne un tenseur (3, H, W) normalisé entre 0 et 1
    ])
    img_tensor = transform(img)

    # Récupération de la hauteur et largeur
    h, w = img_tensor.shape[1:]

    # Conversion en tableau NumPy (H, W, 3) pour affichage
    img_array = img_tensor.permute(1, 2, 0).numpy()

    # --- Création d'une grille normalisée des coordonnées des pixels ---
    x_coords = torch.linspace(0, 1, w)
    y_coords = torch.linspace(0, 1, h)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    # Flatten de la grille pour former les entrées du MLP : shape (H*W, 2)
    inputs = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)

    # Extraction des valeurs RGB comme sorties cibles : shape (H*W, 3)
    outputs = img_tensor.view(3, -1).permute(1, 0)

    return img_array, inputs, outputs
url.help = fPrintDoc(url)

def reshape(img_array, array):
    """
    Reshape un tenseur plat de prédiction en image (H, W, 3) en utilisant
    les dimensions de l’image originale.

    Parameters
    ----------
    img_array : np.ndarray of shape (H, W, 3)
        Image originale servant de référence pour récupérer la hauteur (H)
        et la largeur (W).
    array : tensor-like or ndarray of shape (H*W, 3)
        Tableau plat contenant les valeurs RGB prédites pour chaque pixel.

    Returns
    -------
    np.ndarray of shape (H, W, 3)
        Image reconstruite à partir du tableau plat.

    Notes
    -----
    - Cette fonction ne modifie pas les valeurs, elle fait uniquement un reshape.
    - Utile après une prédiction de type MLP qui renvoie un tableau (N, 3).
    """

    # Récupération de la hauteur et largeur à partir de l’image originale
    h, w = img_array.shape[:2]

    # Reconstruction en image RGB
    return array.reshape(h, w, 3)
reshape.help = fPrintDoc(reshape)

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
    *nets : *MLP
        Un ou plusieurs réseaux.

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
    all_losses = [[loss for loss in net.losses] for net in nets]
    if max(len(lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max(len(lst) for lst in all_losses)
    ax_loss.set_xlim(1, lenlosses)
    ax_loss.set_yscale('log', nonpositive='mask')
    

    # --- Boucle sur chaque réseau pour afficher l'erreur et les pertes ---
    for k, net in enumerate(nets):
        # Subplot pour l'erreur absolue
        ax = fig.add_subplot(gs[1, k])
        ax.axis('off')
        ax.set_title(net.name)

        # Prédiction et reconstruction de l'image
        pred_img = net(inputs).reshape(h, w, 3)

        # Tracé des pertes cumulées
        ax_loss.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)

        # Affichage de l'erreur absolue
        ax.imshow(np.abs(img_array - pred_img))
    ax_loss.legend()
    
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
    *nets : *MLP
        Un ou plusieurs réseaux.
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
    all_losses = [[loss for loss in net.losses] for net in nets]
    if max(len(lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max(len(lst) for lst in all_losses)
    ax_loss.set_xlim(1, lenlosses)

    # --- Boucle sur chaque réseau pour afficher les prédictions et pertes ---
    for k, net in enumerate(nets):
        # Subplot pour l'image reconstruite
        ax = fig.add_subplot(gs[1, k])
        ax.axis('off')
        ax.set_title(net.name)

        # Prédiction et reconstruction de l'image
        pred_img = net(inputs).reshape(h, w, 3)

        # Tracé des pertes cumulées
        ax_loss.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)
        ax_loss.set_yscale('log', nonpositive='mask')
    
        # Affichage de l'image prédite
        ax.imshow(pred_img)
    ax_loss.legend()

    # --- Affichage final ---
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()
    plt.show()
plot.help = fPrintDoc(plot)