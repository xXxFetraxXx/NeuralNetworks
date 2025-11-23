from .Dependances import *

def image_from_url(url, img_size=256):
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
image_from_url.help = fPrintDoc(image_from_url)

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