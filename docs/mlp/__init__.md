# Initialisation du réseau MLP

Initialise un réseau MLP avec options avancées : encodage Fourier,  
normalisation, choix d’optimiseur et de fonction de perte, et compilation.

---

## Parameters

| Parameter       | Type       | Optional | Description |
|-----------------|------------|----------|-------------|
| `layers`        | list[int]  | Yes      | Dimensions successives du réseau (entrée → couches cachées → sortie). Default: `[1, 1, 1]` |
| `learning_rate` | float      | Yes      | Taux d’apprentissage pour l’optimiseur. Default: `1e-3` |
| `Fourier`       | bool       | Yes      | Si True, applique un encodage RFF (Random Fourier Features) sur les entrées. Default: `True` |
| `optimizer`     | str        | Yes      | Nom de l’optimiseur à utiliser (doit être présent dans `Optim_list`). Default: `"ADAM"` |
| `criterion`     | str        | Yes      | Nom de la fonction de perte à utiliser (doit être présent dans `Criterion_list`). Default: `"MSE"` |
| `normalizer`    | str        | Yes      | Type de normalisation / activation à appliquer entre les couches cachées. Default: `"Relu"` |
| `name`          | str        | Yes      | Nom du réseau (pour identification et affichage). Default: `"Net"` |
| `Iscompiled`    | bool       | Yes      | Si True, compile le modèle avec `torch.compile` pour accélérer l’inférence. Default: `True` |

---

## Attributes

- `losses : list` — Historique des pertes durant l’entraînement  
- `layers : list[int]` — Dimensions du réseau après ajustement pour encodage Fourier  
- `encoding : nn.Module` — Module appliquant l’encodage des entrées (RFF ou identité)  
- `norm : nn.Module` — Normalisation / activation utilisée entre les couches cachées  
- `criterion : nn.Module` — Fonction de perte PyTorch sur GPU  
- `model : nn.Sequential` — MLP complet construit dynamiquement  
- `optimizer : torch.optim.Optimizer` — Optimiseur associé au MLP  
- `name : str` — Nom du réseau