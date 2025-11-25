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

---

## Parameters

| Parameter      | Type          | Optional | Description                                                                                                                                                |
|----------------|---------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `layers`       | list[int]     | Yes      | Dimensions successives du réseau (entrée → couches cachées → sortie). Exemple : `[in_features, hidden1, hidden2, ..., out_features]`. Default: `[1, 1, 1]` |
| `learning_rate`| float         | Yes      | Taux d’apprentissage pour l’optimiseur. Default: `1e-3`                                                                                                    |
| `Fourier`      | bool          | Yes      | Si True, applique un encodage Fourier gaussien (RFF) sur les entrées. Default: `True`                                                                      |
| `optim`        | str           | Yes      | Nom de l’optimiseur à utiliser (doit exister dans `optims()`). Default: `"Adam"`                                                                           |
| `crit`         | str           | Yes      | Fonction de perte à utiliser (doit exister dans `crits()`). Default: `"MSE"`                                                                               |
| `norm`         | str           | Yes      | Type de normalisation / activation pour les couches cachées (ex: `"Relu"`). Default: `"Relu"`                                                              |
| `name`         | str           | Yes      | Nom du réseau pour identification ou affichage. Default: `"Net"`                                                                                           |
| `Iscompiled`   | bool          | Yes      | Si True, compile le modèle via `torch.compile` pour accélérer l’inférence. Default: `True`                                                                 |

---

## Attributes

- `losses : list[torch.Tensor]` — Historique des pertes cumulées lors de l'entraînement  
- `layers : list[int]` — Dimensions du réseau, ajustées si encodage Fourier actif  
- `encoding : nn.Module` — Module appliquant l'encodage des entrées (RFF ou identity)  
- `norm : nn.Module` — Normalisation ou activation utilisée dans les couches cachées  
- `crit : nn.Module` — Fonction de perte PyTorch sur le device spécifié  
- `model : nn.Sequential` — MLP complet construit dynamiquement  
- `optim : torch.optim.Optimizer` — Optimiseur associé au MLP  
- `name : str` — Nom du réseau  

---

## Methods

- `plot(inputs, img_array)` — Affiche l’image originale, l’image prédite et la courbe des pertes  
- `train(inputs, outputs, num_epochs=1500, batch_size=1024)` — Entraîne le MLP avec mini-batchs et AMP, stocke les pertes  
- `params()` — Retourne tous les poids du MLP sous forme de liste d’`ndarray`  
- `neurons()` — Retourne tous les biais du MLP sous forme de liste d’`ndarray`  
- `nb_params()` — Calcule le nombre total de paramètres (poids uniquement) du réseau  

---

## Objets et dictionnaires

# **norms()**

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

# **crits()**

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

# **optims()**

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

## Notes

- La classe supporte un entraînement sur GPU via `device`  
- Les fonctions de visualisation utilisent matplotlib et visualtorch  
- Les sorties sont compatibles avec des images normalisées entre 0 et 1  
- Le suivi des pertes permet d’afficher l’évolution du training loss