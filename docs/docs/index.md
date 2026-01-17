# NeuralNetworks Module

Module complet pour la création et l'entraînement de [MultiLayer Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP)  
avec encodage optionnel [Fourier Features](https://en.wikipedia.org/wiki/Random_feature#Random_Fourier_feature) et gestion automatique des pertes.

---

## **Classes**

### **MLP**

Cette classe fournit :

- Un [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) entièrement configurable (dimensions, activation).
- Option d'encodage [Fourier Features](https://en.wikipedia.org/wiki/Random_feature#Random_Fourier_feature) sur les entrées.

---

#### **Paramètres**

| **Paramètres**       | **Type**                                                                                         | **Optionnel** | **Description**                                                                |
|----------------------|--------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------|
| `input_size`         | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | Taille des données en entrée au réseau. Default: `1`                           |
| `output_size`        | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | Taille des données en sortie au réseau. Default: `1`                           |
| `hidden_layers`      | [`list[int]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)   | Oui           | Dimensions successives des couches intermédiaires du réseau. Default: `[1]`    |
| `sigmas`             | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Oui           | Liste de sigma pour encodages RFF. Si None : passthrough. Default: `None`      |
| `fourier_input_size` | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | WIP. Default: `2`                                                              |
| `nb_fourier`         | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | Nombre de fréquences utilisées pour les Fourier Features. Default: `8`         |
| `norm`               | [`norm`](#norms-norms)                                                                           | Oui           | Type de normalisation / activation pour les couches cachées. Default: `'Relu'` |
| `name`               | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  | Oui           | Nom du réseau pour identification ou affichage. Default: `'Net'`               |

#### **Attributs**

- `losses : list[float]`    — Historique des pertes cumulées lors de l'entraînement  
- `learnings : list[float]` — Historique des taux d'apprentissage utilisées lors de l'entraînement  
- `model : nn.Sequential`   — MLP complet construit dynamiquement 
- `name : str`              — Nom du réseau

| **Attributs** | **Type**                                                                                         | **Description**                                                                |
|---------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `losses`      | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Historique des pertes cumulées lors de l'entraînement                          |
| `learnings`   | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Historique des taux d'apprentissage utilisées lors de l'entraînement           |
| `model`       | [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)       | MLP complet construit dynamiquement                                            |
| `name`        | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  | Nom du réseau                                                                  |

---

### **Trainer**

Cette classe fournit :

- Méthode pour entraîner des réseaux avec mini-batchs et [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

#### **Paramètres**

| **Paramètres** | **Type**                                                                                        | **Optionnel** | **Description**                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------|
| `*nets`        | [`MLP`](#mlp-mlp)                                                                               | Non           | Réseaux pour lesquels le trainer va entrainer.                                                                  |
| `inputs`       | [`numpy.array(list[float])`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) | Non           | Données en entrée au réseau.                                                                                    |
| `outputs`      | [`numpy.array(list[float])`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) | Non           | Données en sortie au réseau.                                                                                    |
| `test_size`    | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Proportion des données à utiliser pendant l'entrainement. Si None : utilise toutes les données. Default: `None` |
| `optim`        | [`optim`](#optims-optims)                                                                       | Oui           | Nom de l’optimiseur à utiliser (doit exister dans `optims()`). Default: `'Adam'`                                |
| `init_lr`      | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Taux d’apprentissage initial pour l’optimiseur. Default: `1e-3`                                                 |
| `crit`         | [`crit`](#crits-crits)                                                                          | Oui           | Fonction de perte à utiliser (doit exister dans `crits()`). Default: `MSE'`                                     |
| `batch_size`   | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)        | Oui           | Taille des minibatchs. Default: `1024`                                                                          |

#### **Trainer.train**

Lancement d'un entrainement avec le trainer définit

| **Paramètres**  | **Type**                                                                                 | **Optionnel** | **Description**                         |
|-----------------|------------------------------------------------------------------------------------------|---------------|-----------------------------------------|
| `num_epochs`    | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           | Nombres d'itérations à effectuer.       |
| `activate_tqdm` | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool)           | Oui           | Utilisation d'une barre de progression. |
  
---

## **Méthodes**

### **losses**

Affiche les résidus en fonction des époques d'entrainement des réseaux.

### **learnings**

Affiche les taux d'apprentissage en fonction des époques d'entrainement des réseaux.

---

## **Dictionnaires**

### **norms**

| **Valeurs**   | **Module PyTorch**                                                                         | **Description**                                                                                           |
|---------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `'ReLU'`      | [`nn.ReLU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)           | Fonction d'activation ReLU classique (Rectified Linear Unit).                                             |
| `'LeakyReLU'` | [`nn.LeakyReLU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) | ReLU avec un petit coefficient pour les valeurs négatives (paramètre `negative_slope`).                   |
| `'ELU'`       | [`nn.ELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html)             | Fonction d'activation ELU (Exponential Linear Unit), qui a une meilleure gestion des valeurs négatives.   |
| `'SELU'`      | [`nn.SELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SELU.html)           | SELU (Scaled Exponential Linear Unit), une version améliorée de l'ELU pour des réseaux auto-normalisants. |
| `'GELU'`      | [`nn.GELU()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)           | GELU (Gaussian Error Linear Unit), une activation probabiliste basée sur une fonction gaussienne.         |
| `'Mish'`      | [`nn.Mish()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html)           | ReLU différentiable en tout points avec passage négatif.                                                  |
| `'Softplus'`  | [`nn.Softplus()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html)   | Fonction d'activation qui approxime ReLU mais de manière lissée.                                          |
| `'Sigmoid'`   | [`nn.Sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)     | Fonction d'activation Sigmoid, qui produit une sortie entre 0 et 1.                                       |
| `'Tanh'`      | [`nn.Tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)           | Fonction d'activation Tanh, avec une sortie dans l'intervalle [-1, 1].                                    |
| `'Hardtanh'`  | [`nn.Hardtanh()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)   | Variante de Tanh, avec des sorties limitées entre une plage spécifiée.                                    |
| `'Softsign'`  | [`nn.Softsign()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softsign.html)   | Fonction d'activation similaire à Tanh mais plus souple, avec des valeurs dans [-1, 1].                   |

---

### **optims**

| **Valeurs**   | **Module PyTorch**                                                                               | **Description**                                                                                       |
|---------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `'Adadelta'`  | [`optim.Adadelta()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)   | Optimiseur basé sur les gradients adaptatifs, sans nécessité de réglage du taux d'apprentissage.      |
| `'Adafactor'` | [`optim.Adafactor()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html) | Optimiseur variant d'Adam avec une mise à jour plus efficace de la mémoire pour de grands modèles.    |
| `'Adam'`      | [`optim.Adam()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)           | Optimiseur utilisant un gradient stochastique adaptatif avec des moyennes mobiles des gradients.      |
| `'AdamW'`     | [`optim.AdamW()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)         | Optimiseur avec une régularisation L2 (weight decay) distincte.                                       |
| `'Adamax'`    | [`optim.Adamax()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adamax.html)       | Optimiseur utilisant une norme infinie pour les gradients, plus stable pour certaines configurations. |
| `'ASGD'`      | [`optim.ASGD()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.ASGD.html)           | Optimiseur utilisé pour de grandes données avec une moyenne des gradients.                            |
| `'NAdam'`     | [`optim.NAdam()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.NAdam.html)         | Optimiseur avec une adaptation des moments de second ordre.                                           |
| `'RAdam'`     | [`optim.RAdam()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RAdam.html)         | Optimiseur qui ajuste dynamiquement les moments pour stabiliser l'entraînement.                       |
| `'RMSprop'`   | [`optim.RMSprop()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)     | Optimiseur utilisant une moyenne mobile des carrés des gradients pour réduire les oscillations.       |
| `'Rprop'`     | [`optim.Rprop()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Rprop.html)         | Optimiseur basé sur les mises à jour des poids indépendantes des gradients.                           |
| `'SGD'`       | [`optim.SGD()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)             | Optimiseur souvent utilisée avec un taux d'apprentissage constant ou ajusté.                          |

---

### **crits**

| **Valeurs**              | **Module PyTorch**                                                                                                       | **Description**                                                       |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `'MSE'`                  | [`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)                                          | Perte utilisée pour les régressions.                                  |
| `'L1'`                   | [`nn.L1Loss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)                                     | Perte utilisée pour la régularisation.                                |
| `'SmoothL1'`             | [`nn.SmoothL1Loss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)                         | Perte moins sensible aux outliers.                                    |
| `'Huber'`                | [`nn.HuberLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)                               | Perte moins affectée par les grands écarts.                           |
| `'CrossEntropy'`         | [`nn.CrossEntropyLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)                 | Perte utilisée pour les problèmes de classification multi-classes.    |
| `'KLDiv'`                | [`nn.KLDivLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)                               | Perte utilisée pour des modèles probabilistes.                        |
| `'PoissonNLL'`           | [`nn.PoissonNLLLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html)                     | Perte utilisée pour la modélisation de comptages.                     |
| `'MultiLabelSoftMargin'` | [`nn.MultiLabelSoftMarginLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html) | Perte utilisée pour les problèmes de classification multi-étiquettes. |
 
---

## **device**

variable principale d'allocation des performances

### **Apple Silicon (macOS)**
- Si le système d'exploitation est macOS (nommé `darwin` dans `platform.system()`), la fonction vérifie si l'accélérateur **Metal Performance Shaders** (MPS) est disponible sur l'appareil.
  - Si MPS est disponible (`torch.backends.mps.is_available()`), l'appareil cible sera défini sur `'mps'` (c'est un équivalent de CUDA pour les appareils Apple Silicon).
  
### **Windows**
- Si le système d'exploitation est Windows, la fonction vérifie d'abord si **CUDA** (NVIDIA) est disponible avec `torch.cuda.is_available()`. Si c'est le cas, le périphérique sera défini sur **CUDA**.
  
### **Linux**
- Si le système d'exploitation est Linux, plusieurs vérifications sont effectuées :
  1. **CUDA** (NVIDIA) : Si `torch.cuda.is_available()` renvoie `True`, le périphérique sera défini sur `'cuda'`.
  2. **ROCm** (AMD) : Si le système supporte **ROCm** via `torch.backends.hip.is_available()`, l'appareil sera défini sur `'cuda'` (ROCm est utilisé pour les cartes AMD dans le cadre de l'API CUDA).
  3. **Intel oneAPI / XPU** : Si le système prend en charge **Intel oneAPI** ou **XPU** via `torch.xpu.is_available()`, le périphérique sera défini sur **XPU**.
  
### **Système non reconnu**
- Si aucune des conditions ci-dessus n'est remplie, la fonction retourne `'cpu'` comme périphérique par défaut.

---