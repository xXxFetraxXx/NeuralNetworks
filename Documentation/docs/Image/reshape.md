# Reconstruction d'image depuis tenseur plat (`reshape_pred_to_image`)

Reshape un tenseur plat de prédiction en image (H, W, 3) en utilisant les dimensions de l’image originale.

---

## Paramètres

| Paramètre   | Type                                | Description |
| ----------- | ----------------------------------- | ----------- |
| `img_array` | np.ndarray (H, W, 3)               | Image originale servant de référence pour récupérer la hauteur (H) et la largeur (W). |
| `array`     | tensor-like ou ndarray (H*W, 3)    | Tableau plat contenant les valeurs RGB prédites pour chaque pixel. |

---

## Retours

| Nom    | Type             | Description |
| ------ | ---------------- | ----------- |
| result | np.ndarray (H, W, 3) | Image reconstruite à partir du tableau plat. |

---

## Notes

- Ne modifie pas les valeurs, fait uniquement un reshape.  
- Utile après une prédiction de type MLP qui renvoie un tableau (N, 3).  