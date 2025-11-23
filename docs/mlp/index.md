# Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) avec encodage optionnel Fourier (RFF),  
suivi automatique des pertes, visualisation et compilation PyTorch.

Cette classe fournit :

- Un MLP entièrement configurable (dimensions, normalisation, activation)  
- Option d'encodage Fourier (Random Fourier Features) sur les entrées  
- Méthodes pour entraîner le réseau avec mini-batchs et AMP (Automatic Mixed Precision)  
- Visualisation de l'architecture via visualtorch  
- Suivi et affichage de la perte d'entraînement  
- Accès aux poids, biais et nombre de paramètres  
- Compilation du modèle via `torch.compile` pour accélérer l'inférence  
- Méthode `__call__` permettant l'utilisation directe comme une fonction (`y = net(x)`)

---

## Parameters

| Parameter      | Type          | Optional | Description |
|----------------|---------------|----------|-------------|
| `layers`       | list[int]     | Yes      | Dimensions successives du réseau (entrée → couches cachées → sortie). Exemple : `[in_features, hidden1, hidden2, ..., out_features]`. Default: `[1, 1, 1]` |
| `learning_rate`| float         | Yes      | Taux d’apprentissage pour l’optimiseur. Default: `1e-3` |
| `Fourier`      | bool          | Yes      | Si True, applique un encodage Fourier gaussien (RFF) sur les entrées. Default: `True` |
| `optimizer`    | str           | Yes      | Nom de l’optimiseur à utiliser (doit exister dans `Optim_list`). Default: `"ADAM"` |
| `criterion`    | str           | Yes      | Fonction de perte à utiliser (doit exister dans `Criterion_list`). Default: `"MSE"` |
| `normalizer`   | str           | Yes      | Type de normalisation / activation pour les couches cachées (ex: `"Relu"`). Default: `"Relu"` |
| `name`         | str           | Yes      | Nom du réseau pour identification ou affichage. Default: `"Net"` |
| `Iscompiled`   | bool          | Yes      | Si True, compile le modèle via `torch.compile` pour accélérer l’inférence. Default: `True` |

---

## Attributes

- `losses : list[torch.Tensor]` — Historique des pertes cumulées lors de l'entraînement  
- `layers : list[int]` — Dimensions du réseau, ajustées si encodage Fourier actif  
- `encoding : nn.Module` — Module appliquant l'encodage des entrées (RFF ou identity)  
- `norm : nn.Module` — Normalisation ou activation utilisée dans les couches cachées  
- `criterion : nn.Module` — Fonction de perte PyTorch sur le device spécifié  
- `model : nn.Sequential` — MLP complet construit dynamiquement  
- `optimizer : torch.optim.Optimizer` — Optimiseur associé au MLP  
- `name : str` — Nom du réseau  

---

## Methods

- `__init__(...)` — Initialise le réseau, configure l’encodage, la fonction de perte et l’optimiseur  
- `__repr__()` — Affiche un schéma visuel du MLP et ses dimensions (avec compression si nécessaire)  
- `__call__(x)` — Applique l’encodage et le MLP sur un input x, retourne la prédiction en `ndarray`  
- `Create_MLP(layers)` — Construit un `nn.Sequential` avec les couches linéaires, activations et normalisations  
- `plot(inputs, img_array)` — Affiche l’image originale, l’image prédite et la courbe des pertes  
- `train(inputs, outputs, num_epochs=1500, batch_size=1024)` — Entraîne le MLP avec mini-batchs et AMP, stocke les pertes  
- `params()` — Retourne tous les poids du MLP sous forme de liste d’`ndarray`  
- `neurons()` — Retourne tous les biais du MLP sous forme de liste d’`ndarray`  
- `nb_params()` — Calcule le nombre total de paramètres (poids uniquement) du réseau  

---

## Notes

- La classe supporte un entraînement sur GPU via `device`  
- Les fonctions de visualisation utilisent matplotlib et visualtorch  
- Les sorties sont compatibles avec des images normalisées entre 0 et 1  
- Le suivi des pertes permet d’afficher l’évolution du training loss