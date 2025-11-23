# NeuralNetworks Module

Module complet pour la création, l'entraînement et la visualisation de Multi-Layer Perceptrons (MLP)  
avec encodage optionnel Fourier, gestion automatique des pertes, compilation Torch et outils  
de traitement d'images pour l'apprentissage sur des images RGB.

---

## Contenu principal

### Classes

#### MLP

Multi-Layer Perceptron (MLP) avec options avancées :

- Encodage Fourier gaussien (RFF) optionnel  
- Stockage automatique des pertes  
- Compilation Torch optionnelle pour accélérer l’inférence  
- Gestion flexible de l’optimiseur, de la fonction de perte et de la normalisation  

**Méthodes principales :**

- `__init__(layers, learning_rate, Fourier, optimizer, criterion, normalizer, name, Iscompiled)`  
  Initialise le réseau avec toutes les options.

- `train(inputs, outputs, num_epochs, batch_size)`  
  Entraîne le MLP sur des données (`inputs → outputs`) en utilisant AMP et mini-batchs.

- `plot(inputs, img_array)`  
  Affiche l'image originale, la prédiction du MLP et la courbe des pertes.

- `__call__(x)`  
  Applique l’encodage puis le MLP pour produire une prédiction.

- `Create_MLP(layers)`  
  Construit le MLP avec normalisation/activation et Sigmoid finale.

- `params()`  
  Retourne tous les poids du MLP (ligne par ligne) sous forme de liste de `numpy.ndarray`.

- `nb_params()`  
  Calcule le nombre total de poids dans le MLP.

- `neurons()`  
  Retourne la liste des biais (neurones) de toutes les couches linéaires.

- `__repr__()`  
  Affiche un schéma visuel du MLP via visualtorch et print des dimensions.

---

### Fonctions utilitaires

- `tensorise(obj)`  
  Convertit un objet array-like ou tensor en `torch.Tensor` float32 sur le device actif.

- `list_to_cpu(cuda_tensors)`  
  Copie une liste de tenseurs CUDA et les transfère sur le CPU.

- `rglen(list)`  
  Renvoie un range correspondant aux indices d'une liste.

- `fPrintDoc(obj)`  
  Crée une fonction lambda qui affiche le docstring d'un objet.

- `image_from_url(url, img_size)`  
  Télécharge une image depuis une URL, la redimensionne et génère :
  - `img_array` : `np.ndarray (H, W, 3)` pour affichage.  
  - `inputs` : tenseur `(H*W, 2)` coordonnées normalisées.  
  - `outputs` : tenseur `(H*W, 3)` valeurs RGB cibles.

---

### Visualisation et comparaison

- `plot(img_array, inputs, *nets)`  
  Affiche pour chaque réseau l'image reconstruite à partir des entrées.

- `compare(img_array, inputs, *nets)`  
  Affiche pour chaque réseau l'erreur absolue entre l'image originale et la prédiction,  
  et trace également les pertes cumulées. Chaque réseau doit posséder :  
  - `encoding(x)` si RFF activé  
  - `model()` retournant un tenseur `(N, 3)`  
  - attribut `losses`

---

### Objets et dictionnaires

- `Norm_list : dict`  
  Contient les modules PyTorch correspondant aux fonctions de normalisation/activation disponibles (ReLU, GELU, Sigmoid, Tanh, etc.)

- `Criterion_list : dict`  
  Contient les fonctions de perte PyTorch disponibles (MSE, L1, SmoothL1, BCE, CrossEntropy, etc.)

- `Optim_list(self, learning_rate)`  
  Retourne un dictionnaire d’optimiseurs PyTorch initialisés avec `self.model.parameters()`.

---

### Device et configuration

- `device`  
  Device par défaut (GPU si disponible, sinon CPU).

---

### Paramètres matplotlib et PyTorch

- Style global pour fond transparent et texte gris  
- Optimisations CUDA activées pour TF32, matmul et convolutions  
- Autograd configuré pour privilégier les performances

---

### Notes générales

- Toutes les méthodes de MLP utilisent les tenseurs sur le device global (CPU ou GPU)  
- Les images doivent être normalisées entre 0 et 1  
- Les fonctions interactives (`plot`, `compare`) utilisent matplotlib en mode interactif  
- Le module est conçu pour fonctionner dans Jupyter et scripts Python classiques

## MLP opbject

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