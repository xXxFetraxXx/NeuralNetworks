# `MLP.train()`

Entraîne le MLP sur des paires (inputs → outputs) en utilisant un schéma de mini-batchs et l'AMP (Automatic Mixed Precision).

---

## Paramètres

| Paramètre     | Type                  | Description                                                          |
| ------------- | --------------------- | -------------------------------------------------------------------- |
| `inputs`      | array-like ou tensor  | Données d'entrée du réseau, de shape `(N, input_dim)`.               |
| `outputs`     | array-like ou tensor  | Cibles associées, de shape `(N, output_dim)`.                        |
| `num_epochs`  | int, optionnel        | Nombre total d'époques d'entraînement. Default: `1500`.              |
| `batch_size`  | int, optionnel        | Taille des mini-batchs utilisés à chaque itération. Default: `1024`. |

---

## Notes

- Utilise `torch.amp.autocast` + `GradScaler` pour un entraînement accéléré en FP16  
- Les pertes par époque sont stockées dans `self.losses`  