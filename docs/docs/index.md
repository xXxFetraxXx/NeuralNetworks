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

- `MLP(layers, learning_rate, Fourier, optim, crit, norm, name, Iscompiled)`  
  Initialise le réseau avec toutes les options.

  Les valeurs possibles de `optim`  sont disponibles avec `optims()` 
  Les valeurs possibles de `crit`  sont disponibles avec `crits()` 
  Les valeurs possibles de `norm`  sont disponibles avec `norms()` 

- `train(inputs, outputs, num_epochs, batch_size)`  
  Entraîne le MLP sur des données (`inputs → outputs`) en utilisant AMP et mini-batchs.

- `plot(inputs, img_array)`  
  Affiche l'image originale, la prédiction du MLP et la courbe des pertes.

- `params()`  
  Retourne tous les poids du MLP (ligne par ligne) sous forme de liste de `numpy.ndarray`.

- `nb_params()`  
  Calcule le nombre total de poids dans le MLP.

- `neurons()`  
  Retourne la liste des biais (neurones) de toutes les couches linéaires.

---

### Fonctions utilitaires

- `tensorise(obj)`  
  Convertit un objet array-like ou tensor en `torch.Tensor` float32 sur le device actif.

- `rglen(list)`  
  Renvoie un range correspondant aux indices d'une liste.

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

---

### Objets et dictionnaires

#### **norms()**

| **Valeurs**         | **Module PyTorch**        | **Description**                                                                                           |
|---------------------|---------------------------|-----------------------------------------------------------------------------------------------------------|
| **"Relu"**          | `nn.ReLU()`               | Fonction d'activation ReLU classique (Rectified Linear Unit).                                             |
| **"LeakyRelu"**     | `nn.LeakyReLU()`          | ReLU avec un petit coefficient pour les valeurs négatives (paramètre `negative_slope`).                   |
| **"ELU"**           | `nn.ELU()`                | Fonction d'activation ELU (Exponential Linear Unit), qui a une meilleure gestion des valeurs négatives.   |
| **"SELU"**          | `nn.SELU()`               | SELU (Scaled Exponential Linear Unit), une version améliorée de l'ELU pour des réseaux auto-normalisants. |
| **"GELU"**          | `nn.GELU()`               | GELU (Gaussian Error Linear Unit), une activation probabiliste basée sur une fonction gaussienne.         |
| **"Sigmoid"**       | `nn.Sigmoid()`            | Fonction d'activation Sigmoid, qui produit une sortie entre 0 et 1.                                       |
| **"Tanh"**          | `nn.Tanh()`               | Fonction d'activation Tanh, avec une sortie dans l'intervalle [-1, 1].                                    |
| **"Hardtanh"**      | `nn.Hardtanh()`           | Variante de Tanh, avec des sorties limitées entre une plage spécifiée.                                    |
| **"PReLU"**         | `nn.PReLU()`              | Parametric ReLU, une version de ReLU où le coefficient `slope` est appris.                                |
| **"RReLU"**         | `nn.RReLU()`              | Randomized ReLU, une version de PReLU avec un paramètre de pente aléatoire pendant l'entraînement.        |
| **"Softplus"**      | `nn.Softplus()`           | Fonction d'activation qui approxime ReLU mais de manière lissée.                                          |
| **"Softsign"**      | `nn.Softsign()`           | Fonction d'activation similaire à Tanh mais plus souple, avec des valeurs dans [-1, 1].                   |

---

#### **crits()**

| **Valeurs**                    | **Module PyTorch**                  | **Description**                                                                                                            |
|--------------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **"MSE"**                      | `nn.MSELoss()`                      | Mean Squared Error Loss, utilisée pour les régressions.                                                                    |
| **"L1"**                       | `nn.L1Loss()`                       | L1 Loss (erreur absolue), souvent utilisée pour la régularisation.                                                         |
| **"SmoothL1"**                 | `nn.SmoothL1Loss()`                 | Smooth L1 Loss, une combinaison de L1 et de MSE, moins sensible aux outliers.                                              |
| **"Huber"**                    | `nn.HuberLoss()`                    | Fonction de perte Huber, une version lissée de L1 et MSE, moins affectée par les grands écarts.                            |
| **"CrossEntropy"**             | `nn.CrossEntropyLoss()`             | Perte de Cross-Entropy, utilisée pour les problèmes de classification multi-classes.                                       |
| **"BCE"**                      | `nn.BCELoss()`                      | Binary Cross-Entropy Loss, utilisée pour les tâches de classification binaire.                                             |
| **"BCEWithLogits"**            | `nn.BCEWithLogitsLoss()`            | BCE Loss combinée avec un calcul de logits, plus stable numériquement que l'utilisation séparée de `Sigmoid` et `BCELoss`. |
| **"KLDiv"**                    | `nn.KLDivLoss()`                    | Perte de divergence de Kullback-Leibler, souvent utilisée pour des modèles probabilistes.                                  |
| **"PoissonNLL"**               | `nn.PoissonNLLLoss()`               | Perte de log-vraisemblance pour une distribution de Poisson, utilisée pour la modélisation de comptages.                   |
| **"MultiLabelSoftMargin"**     | `nn.MultiLabelSoftMarginLoss()`     | Perte utilisée pour les problèmes de classification multi-étiquettes.                                                      |

---

#### **optims()**

| **Valeurs**         | **Module PyTorch**                   | **Description**                                                                                                                    |
|---------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **"Adadelta"**      | `optim.Adadelta()`                   | Optimiseur Adadelta, basé sur les gradients adaptatifs, sans nécessité de réglage du taux d'apprentissage.                         |
| **"Adafactor"**     | `optim.Adafactor()`                  | Optimiseur Adafactor, variant d'Adam avec une mise à jour plus efficace de la mémoire pour de grands modèles.                      |
| **"Adam"**          | `optim.Adam()`                       | Optimiseur Adam, utilisant un gradient stochastique adaptatif avec des moyennes mobiles des gradients et des carrés des gradients. |
| **"AdamW"**         | `optim.AdamW()`                      | Optimiseur Adam avec une régularisation L2 (weight decay) distincte, plus efficace que `Adam` avec `weight_decay`.                 |
| **"Adamax"**        | `optim.Adamax()`                     | Version d'Adam utilisant une norme infinie pour les gradients, plus stable pour certaines configurations.                          |
| **"ASGD"**          | `optim.ASGD()`                       | Optimiseur ASGD (Averaged Stochastic Gradient Descent), utilisé pour de grandes données avec une moyenne des gradients.            |
| **"NAdam"**         | `optim.NAdam()`                      | Optimiseur NAdam, une version améliorée d'Adam avec une adaptation des moments de second ordre.                                    |
| **"RAdam"**         | `optim.RAdam()`                      | Optimiseur RAdam, une version robuste de l'Adam qui ajuste dynamiquement les moments pour stabiliser l'entraînement.               |
| **"RMSprop"**       | `optim.RMSprop()`                    | Optimiseur RMSprop, utilisant une moyenne mobile des carrés des gradients pour réduire les oscillations.                           |
| **"Rprop"**         | `optim.Rprop()`                      | Optimiseur Rprop, basé sur les mises à jour des poids indépendantes des gradients.                                                 |
| **"SGD"**           | `optim.SGD()`                        | Descente de gradient stochastique classique, souvent utilisée avec un taux d'apprentissage constant ou ajusté.                     |

---

### Device et configuration

#### **Apple Silicon (macOS)**
- Si le système d'exploitation est macOS (nommé `darwin` dans `platform.system()`), la fonction vérifie si l'accélérateur **Metal Performance Shaders** (MPS) est disponible sur l'appareil.
  - Si MPS est disponible (`torch.backends.mps.is_available()`), l'appareil cible sera défini sur **MPS** (c'est un équivalent de CUDA pour les appareils Apple Silicon).
  
#### **Windows**
- Si le système d'exploitation est Windows, la fonction vérifie d'abord si **CUDA** (NVIDIA) est disponible avec `torch.cuda.is_available()`. Si c'est le cas, le périphérique sera défini sur **CUDA**.
  
#### **Linux**
- Si le système d'exploitation est Linux, plusieurs vérifications sont effectuées :
  1. **CUDA** (NVIDIA) : Si `torch.cuda.is_available()` renvoie `True`, le périphérique sera défini sur **CUDA**.
  2. **ROCm** (AMD) : Si le système supporte **ROCm** via `torch.backends.hip.is_available()`, l'appareil sera défini sur **CUDA** (ROCm est utilisé pour les cartes AMD dans le cadre de l'API CUDA).
  3. **Intel oneAPI / XPU** : Si le système prend en charge **Intel oneAPI** ou **XPU** via `torch.xpu.is_available()`, le périphérique sera défini sur **XPU**.
  
#### **Système non reconnu**
- Si aucune des conditions ci-dessus n'est remplie, la fonction retourne **CPU** comme périphérique par défaut.

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