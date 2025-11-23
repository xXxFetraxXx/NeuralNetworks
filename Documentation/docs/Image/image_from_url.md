# Préparation d'image pour MLP (`image_from_url`)

Télécharge une image depuis une URL, la redimensionne et prépare les données pour l'entraînement d'un MLP pixel-wise.

Cette fonction retourne :

- **img_array** : image RGB sous forme de tableau NumPy (H, W, 3), pour affichage.  
- **inputs** : coordonnées normalisées (x, y) de chaque pixel, sous forme de tenseur (H*W, 2).  
- **outputs** : valeurs RGB cibles pour chaque pixel, sous forme de tenseur (H*W, 3).
---

## Paramètres

| Paramètre  | Type    | Description |
| ---------- | ------- | ----------- |
| `url`      | str     | URL de l'image à télécharger. |
| `img_size` | int     | Taille finale carrée de l'image (img_size x img_size). Par défaut 256. |

---

## Retours

| Nom         | Type | Description |
| ----------- | ---- | ----------- |
| `img_array` | numpy.ndarray (H, W, 3) | Image sous forme de tableau NumPy, valeurs normalisées entre 0 et 1. |
| `inputs`    | torch.Tensor (H*W, 2)    | Coordonnées normalisées des pixels pour l'entrée du MLP. |
| `outputs`   | torch.Tensor (H*W, 3)    | Valeurs RGB cibles pour chaque pixel, pour la sortie du MLP. |

---

## Notes

- Utilise `PIL` pour le traitement de l'image et `torchvision.transforms` pour la conversion en tenseur normalisé.  
- Les coordonnées sont normalisées dans [0, 1] pour une utilisation optimale avec des MLP utilisant Fourier Features ou activations standard.  
- Les tenseurs `inputs` et `outputs` sont prêts à être envoyés sur GPU si nécessaire.  
