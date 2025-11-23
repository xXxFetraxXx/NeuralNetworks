# Affichage de l’image et de la perte (`plot`)

Affiche côte à côte :  
- l’image originale  
- l’image prédite par le MLP  
- l’évolution de la fonction de perte (loss) au cours de l’entraînement

---

## Paramètres

| Paramètre   | Type                  | Description |
| ----------- | -------------------- | ----------- |
| `inputs`    | array-like ou torch.Tensor | Tableau des coordonnées (ou features) servant d’entrée au réseau. Doit correspondre à la grille permettant de reconstruire l’image. |
| `img_array` | np.ndarray            | Image originale sous forme de tableau `(H, W, 3)` utilisée comme référence. |

---

## Notes

Cette méthode :  
- Tensorise les entrées puis les encode avant passage dans le MLP  
- Reshape la sortie du modèle pour retrouver la forme `(H, W, 3)`  
- Trace également la courbe de pertes stockée dans `self.losses`
