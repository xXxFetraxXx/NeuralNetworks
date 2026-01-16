# NeuralNetworks Module

Module complet pour la création, l'entraînement et la visualisation de Multi-Layer Perceptrons (MLP)  
avec encodage optionnel Fourier, gestion automatique des pertes, compilation Torch et outils  
de traitement d'images pour l'apprentissage sur des images RGB.

---

## Contenu principal

### Classes

#### `MLP`

Multi-Layer Perceptron (MLP) avec encodage optionnel Fourier (RFF),  
suivi automatique des pertes, visualisation et compilation PyTorch.

Cette classe fournit :

- Un MLP entièrement configurable (dimensions, normalisation, activation)  
- Option d'encodage Fourier (Random Fourier Features) sur les entrées  
- Méthodes pour entraîner le réseau avec mini-batchs et AMP (Automatic Mixed Precision)  
- Visualisation de l'architecture via visualtorch  
- Suivi et affichage de la perte d'entraînement  
- Accès aux poids, biais et nombre de paramètres  

---

##### Parameters

| Parameter            | Type          | Optional | Description                                                                                   |
|----------------------|---------------|----------|-----------------------------------------------------------------------------------------------|
| `input_size`         | int           | Yes      | Taille des données en entrée au réseau. Default: `1`                                   |
| `output_size`        | int           | Yes      | Taille des données en sortie au réseau. Default: `1`                                   |
| `hidden_layers`      | list[int]     | Yes      | Dimensions successives des couches intermédiaires du réseau. Default: `[1]`                   |
| `sigmas`             | list[float]   | Yes      | Liste de sigma pour encodages RFF. Si None : passthrough. Default: `None`                     |
| `fourier_input_size` | int           | Yes      | WIP. Default: `2`                                                                             |
| `nb_fourier`         | int           | Yes      | Nombre de fréquences utilisées pour les Fourier Features. Default: `8`                        |
| `norm`               | str           | Yes      | Type de normalisation / activation pour les couches cachées (ex: `"Relu"`). Default: `"Relu"` |
| `name`               | str           | Yes      | Nom du réseau pour identification ou affichage. Default: `"Net"`                              |

---

##### Attributes

- `losses : list[float]` — Historique des pertes cumulées lors de l'entraînement  
- `learnings : list[float]` — Historique des taux d'apprentissage utilisées lors de l'entraînement  
- `model : nn.Sequential` — MLP complet construit dynamiquement 
- `name : str` — Nom du réseau

---

#### `Trainer`

---

##### Parameters

| Parameter    | Type            | Optional | Description                                                                                                     |
|--------------|-----------------|----------|-----------------------------------------------------------------------------------------------------------------|
| `*nets`      | *MLP            | No       | Réseaux pour lesquels le trainer va entrainer.                                                                  |
| `inputs`     | np.array(float) | No       | Données en entrée au réseau.                                                                                    |
| `outputs`    | np.array(float) | No       | Données en sortie au réseau.                                                                                    |
| `test_size`  | float           | Yes      | Proportion des données à utiliser pendant l'entrainement. Si None : utilise toutes les données. Default: `None` |
| `optim`      | str             | Yes      | Nom de l’optimiseur à utiliser (doit exister dans `optims()`). Default: `"Adam"`                                |
| `init_lr`    | float           | Yes      | Taux d’apprentissage initial pour l’optimiseur. Default: `1e-3`                                                 |
| `crit`       | str             | Yes      | Fonction de perte à utiliser (doit exister dans `crits()`). Default: `"MSE"`                                    |
| `batch_size` | float           | Yes      | Taille des minibatchs. Default: `1024`                                                                          |

---

##### `Trainer.train`

Lancement d'un entrainement avec le trainer définit

| Parameter       | Type    | Optional | Description                             |
|-----------------|---------|----------|-----------------------------------------|
| `num_epochs`    | int     | Yes      | Nombres d'itérations à effectuer.       |
| `activate_tqdm` | boolean | Yes      | Utilisation d'une barre de progression. |
  
---

### Dictionnaires

#### `norms()`

| **Valeurs**         | **Module PyTorch**                                                                         | **Description**                                                                                           |
|---------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **"ReLU"**          | [`nn.ReLU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)           | Fonction d'activation ReLU classique (Rectified Linear Unit).                                             |
| **"LeakyReLU"**     | [`nn.LeakyReLU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) | ReLU avec un petit coefficient pour les valeurs négatives (paramètre `negative_slope`).                   |
| **"ELU"**           | [`nn.ELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html)             | Fonction d'activation ELU (Exponential Linear Unit), qui a une meilleure gestion des valeurs négatives.   |
| **"SELU"**          | [`nn.SELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SELU.html)           | SELU (Scaled Exponential Linear Unit), une version améliorée de l'ELU pour des réseaux auto-normalisants. |
| **"GELU"**          | [`nn.GELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)           | GELU (Gaussian Error Linear Unit), une activation probabiliste basée sur une fonction gaussienne.         |
| **"Mish"**          | [`nn.Mish()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html)           | ReLU différentiable en tout points avec passage négatif.                                                  |
| **"Softplus"**      | [`nn.Softplus()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html)   | Fonction d'activation qui approxime ReLU mais de manière lissée.                                          |
| **"Sigmoid"**       | [`nn.Sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)     | Fonction d'activation Sigmoid, qui produit une sortie entre 0 et 1.                                       |
| **"Tanh"**          | [`nn.Tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)           | Fonction d'activation Tanh, avec une sortie dans l'intervalle [-1, 1].                                    |
| **"Hardtanh"**      | [`nn.Hardtanh()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)   | Variante de Tanh, avec des sorties limitées entre une plage spécifiée.                                    |
| **"Softsign"**      | [`nn.Softsign()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softsign.html)   | Fonction d'activation similaire à Tanh mais plus souple, avec des valeurs dans [-1, 1].                   |

---

#### `crits()`

| **Valeurs**                    | **Module PyTorch**                  | **Description**                                                                                                            |
|--------------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **"MSE"**                      | [`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) | Mean Squared Error Loss, utilisée pour les régressions.                        |
| **"L1"**                       | `nn.L1Loss()`                       | L1 Loss (erreur absolue), souvent utilisée pour la régularisation.                                                         |
| **"SmoothL1"**                 | `nn.SmoothL1Loss()`                 | Smooth L1 Loss, une combinaison de L1 et de MSE, moins sensible aux outliers.                                              |
| **"Huber"**                    | `nn.HuberLoss()`                    | Fonction de perte Huber, une version lissée de L1 et MSE, moins affectée par les grands écarts.                            |
| **"CrossEntropy"**             | `nn.CrossEntropyLoss()`             | Perte de Cross-Entropy, utilisée pour les problèmes de classification multi-classes.                                       |
| **"KLDiv"**                    | `nn.KLDivLoss()`                    | Perte de divergence de Kullback-Leibler, souvent utilisée pour des modèles probabilistes.                                  |
| **"PoissonNLL"**               | `nn.PoissonNLLLoss()`               | Perte de log-vraisemblance pour une distribution de Poisson, utilisée pour la modélisation de comptages.                   |
| **"MultiLabelSoftMargin"**     | `nn.MultiLabelSoftMarginLoss()`     | Perte utilisée pour les problèmes de classification multi-étiquettes.                                                      |

---

#### `optims()`

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

### `device`

variable principale d'allocation des performances

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