"""
NeuralNetworks Module
====================

Module complet pour la création, l'entraînement et la visualisation de Multi-Layer Perceptrons (MLP)
avec encodage optionnel Fourier, gestion automatique des pertes, compilation Torch et outils
de traitement d'images pour l'apprentissage sur des images RGB.

Contenu principal
-----------------

Classes
-------

MLP
   Multi-Layer Perceptron (MLP) avec options avancées :
   - Encodage Fourier gaussien (RFF) optionnel.
   - Stockage automatique des pertes.
   - Compilation Torch optionnelle pour accélérer l’inférence.
   - Gestion flexible de l’optimiseur, de la fonction de perte et de la normalisation.

   Méthodes principales :
   - __init__(layers, learning_rate, Fourier, optimizer, criterion, normalizer, name, Iscompiled)
     Initialise le réseau avec toutes les options.
   - train(inputs, outputs, num_epochs, batch_size)
     Entraîne le MLP sur des données (inputs → outputs) en utilisant AMP et mini-batchs.
   - plot(inputs, img_array)
     Affiche l'image originale, la prédiction du MLP et la courbe des pertes.
   - __call__(x)
     Applique l’encodage puis le MLP pour produire une prédiction.
   - Create_MLP(layers)
     Construit le MLP avec normalisation/activation et Sigmoid finale.
   - params()
     Retourne tous les poids du MLP (ligne par ligne) sous forme de liste de numpy.ndarray.
   - nb_params()
     Calcule le nombre total de poids dans le MLP.
   - neurons()
     Retourne la liste des biais (neurones) de toutes les couches linéaires.
   - __repr__()
     Affiche un schéma visuel du MLP via visualtorch et print des dimensions.

Fonctions utilitaires
--------------------

tensorise(obj)
   Convertit un objet array-like ou tensor en torch.Tensor float32 sur le device actif.

list_to_cpu(cuda_tensors)
   Copie une liste de tenseurs CUDA et les transfère sur le CPU.

rglen(list)
   Renvoie un range correspondant aux indices d'une liste.

fPrintDoc(obj)
   Crée une fonction lambda qui affiche le docstring d'un objet.

image_from_url(url, img_size)
   Télécharge une image depuis une URL, la redimensionne et génère :
   - img_array : np.ndarray (H, W, 3) pour affichage.
   - inputs : tenseur (H*W, 2) coordonnées normalisées.
   - outputs : tenseur (H*W, 3) valeurs RGB cibles.

Visualisation et comparaison
----------------------------

plot(img_array, inputs, *nets)
   Affiche pour chaque réseau l'image reconstruite à partir des entrées.

compare(img_array, inputs, *nets)
   Affiche pour chaque réseau l'erreur absolue entre l'image originale et la prédiction,
   et trace également les pertes cumulées. Chaque réseau doit posséder :
   - encoding(x) si RFF activé
   - model() retournant un tenseur (N, 3)
   - attribute losses

Objets et dictionnaires
-----------------------

Norm_list : dict
   Contient les modules PyTorch correspondant aux fonctions de normalisation/activation
   disponibles (ReLU, GELU, Sigmoid, Tanh, etc.)

Criterion_list : dict
   Contient les fonctions de perte PyTorch disponibles (MSE, L1, SmoothL1, BCE, CrossEntropy, etc.)

Optim_list(self, learning_rate)
   Retourne un dictionnaire d’optimiseurs PyTorch initialisés avec `self.model.parameters()`.

Device et configuration
-----------------------

device
   Device par défaut (GPU si disponible, sinon CPU).

Paramètres matplotlib et PyTorch
   - Style global pour fond transparent et texte gris.
   - Optimisations CUDA activées pour TF32, matmul et convolutions.
   - Autograd configuré pour privilégier les performances.

Notes générales
---------------

- Toutes les méthodes de MLP utilisent les tenseurs sur le device global (CPU ou GPU).
- Les images doivent être normalisées entre 0 et 1.
- Les fonctions interactives (plot, compare) utilisent matplotlib en mode interactif.
- Le module est conçu pour fonctionner dans Jupyter et scripts Python classiques.
"""

# Import des dépendances et utilitaires globaux (device, settings, tensorise, etc.)
from .Dependances import norms, crits, optims, rglen, device, pi, e, tensorise, norm_list, crit_list, optim_list

# Fonctions de chargement/preprocessing des images
from .Image import image_from_url

# Fonctions d'affichage : reconstruction, comparaison, courbes de pertes
from .Plot import compare, plot, losses, train

# Modèle MLP principal + fonction d'entraînement associée
from .MLP import MLP

__version__ = "0.1.10"