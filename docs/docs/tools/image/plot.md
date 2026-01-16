# `plot`

Affiche, pour chaque réseau, l’image reconstruite à partir de ses prédictions.

---

## Paramètres

| Paramètre    | Type                 | Description                                                                                          |
| ------------ | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `img_array`  | np.ndarray (H, W, 3) | Image originale, utilisée pour connaître les dimensions de reconstruction.                           |
| `inputs`     | tensor-like (H*W, 2) | Coordonnées normalisées des pixels correspondant à chaque point de l'image.                          |
| `*nets`      | MLP ou liste de MLP  | Un ou plusieurs réseaux possédant les méthodes `.encoding()` et `.model()`, et l’attribut `.losses`. |

---

## Notes

- Affiche la prédiction brute de chaque réseau.  
- Les pertes cumulées sont également tracées.  
- Utilise matplotlib en mode interactif.
