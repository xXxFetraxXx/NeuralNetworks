# Entraînement de MLP (`train_mlps`)

Entraîne un ou plusieurs MLP sur des paires (inputs, outputs) avec gestion optionnelle de l'affichage interactif.

Affiche dynamiquement si `img_array` est fourni :
- L'image originale (référence)  
- Les prédictions des MLP  
- L'évolution des pertes au fil des époques

---

## Paramètres

| Paramètre    | Type                           | Description                                                           |
| ------------ | ------------------------------ | --------------------------------------------------------------------- |
| `inputs`     | array-like                     | Entrées du ou des MLP (shape: [n_samples, n_features]).               |
| `outputs`    | array-like                     | Sorties cibles correspondantes (shape: [n_samples, output_dim]).      |
| `num_epochs` | int, optional                  | Nombre d’époques pour l’entraînement (default=1500).                  |
| `batch_size` | int, optional                  | Taille des mini-batchs pour la descente de gradient (default=1024).   |
| `*nets`      | MLP ou liste de MLP            | Un ou plusieurs objets MLP à entraîner.                               |
| `img_array`  | np.ndarray (H, W, 3), optional | Image de référence pour visualisation des prédictions (default=None). |

---

## Notes

- Les MLP sont entraînés indépendamment mais avec le même ordre aléatoire d'échantillons.  
- Utilise `torch.amp.GradScaler` pour l'entraînement en FP16.  
- La visualisation interactive utilise `clear_output()` et `plt.ion()`.
