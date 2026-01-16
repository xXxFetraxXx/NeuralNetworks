# `compare`

Affiche, pour chaque réseau, l’erreur absolue entre l’image originale et l’image reconstruite par le réseau.

Chaque réseau doit posséder :  
- une méthode `encoding(x)` (si RFF activé),  
- un module `model` retournant un tenseur de shape (N, 3),  
- une reconstruction compatible avec (H, W, 3).  

---

## Paramètres

| Paramètre   | Type                        | Description                                                                                          |
| ----------- | --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `img_array` | np.ndarray (H, W, 3)        | Image originale servant de référence.                                                                |
| `inputs`    | tensor-like (H*W, 2)        | Coordonnées normalisées des pixels correspondant à chaque point de l'image.                          |
| `nets`      | MLP ou liste de MLP         | Un ou plusieurs réseaux possédant les méthodes `.encoding()` et `.model()`, et l’attribut `.losses`. |

---

## Notes

- Affiche la différence absolue entre l’image originale et la prédiction du réseau.  
- Les pertes cumulées sont également tracées pour chaque réseau.  
- Utilise matplotlib en mode interactif.  