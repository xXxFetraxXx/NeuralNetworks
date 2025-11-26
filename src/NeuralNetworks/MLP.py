# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .Dependances import * 

class MLP():
    """
    Multi-Layer Perceptron (MLP) avec encodage optionnel Fourier (RFF),
    suivi automatique des pertes, visualisation et compilation PyTorch.

    Cette classe fournit :
    - Un MLP entièrement configurable (dimensions, normalisation, activation),
    - Option d'encodage Fourier (Random Fourier Features) sur les entrées,
    - Méthodes pour entraîner le réseau avec mini-batchs et AMP (Automatic Mixed Precision),
    - Visualisation de l'architecture via visualtorch,
    - Suivi et affichage de la perte d'entraînement,
    - Accès aux poids, biais et nombre de paramètres,
    - Compilation du modèle via `torch.compile` pour accélérer l'inférence,
    - Méthode `__call__` permettant l'utilisation directe comme une fonction (`y = net(x)`).

    Parameters
    ----------
    layers : list[int], optional
        Dimensions successives du réseau (entrée → couches cachées → sortie).
        Exemple : [in_features, hidden1, hidden2, ..., out_features].
        Default: [1, 1, 1]
    learning_rate : float, optional
        Taux d’apprentissage pour l’optimiseur.
        Default: 1e-3
    Fourier : bool, optional
        Si True, applique un encodage Fourier gaussien (RFF) sur les entrées.
        Default: True
    optim : str, optional
        Nom de l’optimiseur à utiliser (doit exister dans `optim_list`).
        Default: "ADAM"
    crit : str, optional
        Fonction de perte à utiliser (doit exister dans `crit_list`).
        Default: "MSE"
    norm : str, optional
        Type de normalisation / activation pour les couches cachées (ex: "Relu").
        Default: "Relu"
    name : str, optional
        Nom du réseau pour identification ou affichage.
        Default: "Net"
    Iscompiled : bool, optional
        Si True, compile le modèle via `torch.compile` pour accélérer l’inférence.
        Default: True

    Attributes
    ----------
    losses : list[torch.Tensor]
        Historique des pertes cumulées lors de l'entraînement.
    layers : list[int]
        Dimensions du réseau, ajustées si encodage Fourier actif.
    encoding : nn.Module
        Module appliquant l'encodage des entrées (RFF ou identity).
    norm : nn.Module
        Normalisation ou activation utilisée dans les couches cachées.
    crit : nn.Module
        Fonction de perte PyTorch sur le device spécifié.
    model : nn.Sequential
        MLP complet construit dynamiquement.
    optim : torch.optim.Optimizer
        Optimiseur associé au MLP.
    name : str
        Nom du réseau.
    
    Methods
    -------
    __init__(...)
        Initialise le réseau, configure l’encodage, la fonction de perte et l’optimiseur.
    __repr__()
        Affiche un schéma visuel du MLP et ses dimensions (avec compression si nécessaire).
    __call__(x)
        Applique l’encodage et le MLP sur un input x, retourne la prédiction en ndarray.
    Create_MLP(layers)
        Construit un nn.Sequential avec les couches linéaires, activations et normalisations.
    plot(inputs, img_array)
        Affiche l’image originale, l’image prédite et la courbe des pertes.
    train(inputs, outputs, num_epochs=1500, batch_size=1024)
        Entraîne le MLP avec mini-batchs et AMP, stocke les pertes.
    params()
        Retourne tous les poids du MLP sous forme de liste d’ndarray.
    neurons()
        Retourne tous les biais du MLP sous forme de liste d’ndarray.
    nb_params()
        Calcule le nombre total de paramètres (poids uniquement) du réseau.

    Notes
    -----
    - La classe supporte un entraînement sur GPU via `device`.
    - Les fonctions de visualisation utilisent matplotlib et visualtorch.
    - Les sorties sont compatibles avec des images normalisées entre 0 et 1.
    - Le suivi des pertes permet d’afficher l’évolution du training loss.
    """

    def __init__(self, layers=[1,1,1], learning_rate=1e-3, Fourier=True,
                 optim="Adam", crit="MSE", norm="Relu",
                 name="Net", Iscompiled=False):
        """
        Initialise un réseau MLP avec options avancées : encodage Fourier,
        normalisation, choix d’optimiseur et de fonction de perte, et compilation.
    
        Parameters
        ----------
        layers : list[int], optional
            Dimensions successives du réseau (entrée → couches cachées → sortie).
            Default: [1, 1, 1]
        learning_rate : float, optional
            Taux d’apprentissage pour l’optimiseur (default: 1e-3).
        Fourier : bool, optional
            Si True, applique un encodage RFF (Random Fourier Features) sur les entrées.
            Default: True
        optim : str, optional
            Nom de l’optimiseur à utiliser (doit être présent dans `optim_list`).
            Default: "Adam"
        crit : str, optional
            Nom de la fonction de perte à utiliser (doit être présent dans `crit_list`).
            Default: "MSE"
        norm : str, optional
            Type de normalisation / activation à appliquer entre les couches cachées.
            Default: "Relu"
        name : str, optional
            Nom du réseau (pour identification et affichage).
            Default: "Net"
        Iscompiled : bool, optional
            Si True, compile le modèle avec `torch.compile` pour accélérer l’inférence.
            Default: True
    
        Attributes
        ----------
        losses : list
            Historique des pertes durant l’entraînement.
        layers : list[int]
            Dimensions du réseau après ajustement pour encodage Fourier.
        encoding : nn.Module
            Module appliquant l’encodage des entrées (RFF ou identité).
        norm : nn.Module
            Normalisation / activation utilisée entre les couches cachées.
        crit : nn.Module
            Fonction de perte PyTorch sur GPU.
        model : nn.Sequential
            MLP complet construit dynamiquement.
        optim : torch.optim.Optimizer
            Optimiseur associé au MLP.
        name : str
            Nom du réseau.
        """
    
        super().__init__()
    
        # --- Initialisation des attributs de base ---
        self.losses, self.layers, self.Fourier = [], layers.copy(), Fourier
        self.name = name
    
        # --- Encodage Fourier (RFF) ou passthrough ---
        if self.Fourier:
            self.encoding = rff.layers.GaussianEncoding(
                sigma=10.0,
                input_size=self.layers[0],
                encoded_size=self.layers[1]
            ).to(device)
            self.layers[0] = self.layers[1] * 2  # chaque entrée est doublée après encodage
        else:
            self.encoding = nn.Identity().to(device)  # passthrough si pas de Fourier
    
        # --- Sélection du normalisateur / activation ---
        self.norm = norm_list.get(norm)
        if self.norm is None:
            print("")
            print (f"{norm} n'est pas reconnu")
            self.norm = norm_list.get("Relu")
            print (f"Retour au paramètre par défaut: 'Relu'")
    
        # --- Fonction de perte ---
        self.crit = crit_list.get(crit)
        if self.crit is None:
            print("")
            print (f"{crit} n'est pas reconnu")
            self.crit = crit_list.get("MSE")
            print (f"Retour au paramètre par défaut: 'MSE'")
        # --- Construction du MLP ---
        self.model = self.Create_MLP(self.layers)
    
        # --- Sélection de l’optimiseur ---
        self.optim = optim_list(self, learning_rate).get(optim)
        if self.optim is None:
            print("")
            print (f"{optim} n'est pas reconnu")
            self.optim = optim_list(self, learning_rate).get("Adam")
            print (f"Retour au paramètre par défaut: 'Adam'")
        
        # --- Compilation optionnelle du modèle pour accélérer l’inférence ---
        if not has_gcc():
            Iscompiled = False
        
        if Iscompiled:
            self.model = torch.compile(
                self.model,
                mode="max-autotune",
                fullgraph=True,
                dynamic=True
            )
    
        # --- Envoi du modèle sur le device GPU / CPU ---
        self.model.to(device)

    def __repr__(self):
        """
        Génère un aperçu visuel du MLP et affiche ses dimensions.
    
        Cette méthode :
        - crée une version éventuellement "compressée" des dimensions du réseau
          (utile lorsque certaines couches dépassent 30 neurones, afin de
          conserver une visualisation lisible),
        - utilise `visualtorch.graph_view` pour afficher un schéma du MLP,
        - imprime la liste réelle des dimensions du réseau,
        - retourne une chaîne indiquant si un redimensionnement a été appliqué.
    
        Notes
        -----
        - Le redimensionnement ne modifie pas le MLP réel. Il ne sert qu'à
          améliorer la lisibilité du graphe affiché.
        - Si Fourier Features sont activées, seule la première dimension est
          recalculée en conséquence.
        """
    
        # Si les couches sont trop grandes, on crée une version réduite
        if max(self.layers) > 30:
            # Mise à l’échelle proportionnelle sur une base max 32
            fakelayers = [int(32 * layer / max(self.layers)) for layer in self.layers]
    
            # Ajustement de la couche d’entrée si encodage Fourier
            fakelayers[0] = (
                int(32 * self.layers[0] / max(self.layers))
                if self.Fourier else self.layers[0]
            )
    
            # La couche de sortie reste intacte
            fakelayers[-1] = self.layers[-1]
    
        else:
            # Sinon, on garde les dimensions réelles
            fakelayers = self.layers

        # Affichage console des dimensions réelles
        print("Tailles réelles :")
        print(str(self.layers))
        print("")
        print("Tailles affichées :")
        print(str(fakelayers))
    
        # --- Visualisation du MLP ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
    
        # Utilisation de visualtorch pour tracer l’architecture
        ax.imshow(
            visualtorch.graph_view(
                self.Create_MLP(fakelayers),
                (1, fakelayers[0]),
                ellipsize_after=34,
                background_fill=(0, 0, 0, 0),
                opacity=255
            )
        )
    
        plt.show()
        return ""

    def __call__(self, x):
        """
        Effectue une inférence complète : encodage éventuel, passage
        dans le MLP, puis retour en numpy.
    
        Cette méthode permet d’utiliser l’objet comme une fonction
        directement : `y = net(x)`.
    
        Paramètres
        ----------
        x : array-like
            Entrée(s) à prédire. Peut être un tableau numpy, une liste,
            ou déjà un tenseur compatible.
    
        Returns
        -------
        np.ndarray
            Sortie du MLP après encodage et propagation avant,
            convertie en tableau numpy sur CPU.
        """
        
        # Inférence sans calcul de gradient (plus rapide et évite la construction du graphe)
        with torch.no_grad():
            return self.model(
                self.encoding(tensorise(x))
            ).cpu().numpy()

    def params(self):
        """
        Retourne la liste de tous les poids (weights) du MLP.
    
        Cette fonction extrait uniquement les matrices de poids des couches
        linéaires, en ignorant les biais.  
        Dans un `nn.Linear`, `parameters()` renvoie dans l'ordre :
            - les poids (indice pair),
            - les biais (indice impair).
    
        La méthode :
        - parcourt les paramètres du réseau,
        - sélectionne uniquement ceux correspondant aux poids,
        - sépare chaque ligne de la matrice de poids,
        - convertit chaque ligne en ndarray CPU détaché.
    
        Returns
        -------
        list[np.ndarray]
            Liste contenant chaque ligne des matrices de poids
            (chaque élément est un vecteur numpy).
        """
    
        list_weights = []
        params = list(self.model.parameters())
    
        # Indices pairs → matrices de poids (indices impairs = biais)
        for i in rglen(params):
            if i % 2 == 0:
                weights = list(params[i])  # Chaque élément = ligne de la matrice W
                for j in rglen(weights):
                    # On convertit ligne par ligne en numpy pour inspection externe
                    list_weights.append(weights[j].detach().cpu().numpy())
    
        return list_weights

    def nb_params(self):
        """
        Calcule le nombre total de paramètres (poids) du MLP.
    
        Cette méthode parcourt les paramètres du modèle et ne compte
        que les **poids** des couches linéaires.  
        Dans un nn.Linear, `parameters()` renvoie dans l'ordre :
            - les poids (indice pair),
            - puis les biais (indice impair).
        On ignore donc les biais dans ce comptage.
    
        Returns
        -------
        int
            Nombre total de paramètres pondérés (weights) dans le MLP.
        """
    
        sum = 0
        params = list(self.model.parameters())
    
        # Les indices pairs correspondent aux matrices de poids
        for i in rglen(params):
            if i % 2 == 0:
                weights = list(params[i])
                # On additionne la longueur de chaque ligne de la matrice de poids
                for j in rglen(weights):
                    sum += len(weights[j])
    
        return sum

    def neurons(self):
        """
        Extrait l'ensemble des neurones (poids) du MLP couche par couche.
    
        Cette méthode parcourt les paramètres du modèle et récupère
        uniquement les poids associés aux biais des couches linéaires.
        Dans un nn.Linear, `parameters()` renvoie successivement :
            - les poids (weights),
            - puis les biais (bias).
        Les indices impairs correspondent donc aux biais.
    
        Returns
        -------
        list of ndarray
            Liste contenant chaque neurone (chaque valeur de biais),
            converti en array NumPy sur CPU.
        """
        
        list_neurons = []
        params = list(self.model.parameters())
    
        # Parcours des paramètres et sélection des biais (indices impairs)
        for i in rglen(params):
            if i % 2 == 1:  # Les biais des nn.Linear
                neurons = list(params[i])
                # Extraction individuelle de chaque neurone
                for j in rglen(neurons):
                    list_neurons.append(neurons[j].detach().cpu().numpy())
    
        return list_neurons

    def Create_MLP(self, layers):
        """
        Construit un Multi-Layer Perceptron (MLP) standard composé :
        - d'une succession de couches Linéaires,
        - suivies d'une normalisation (self.norm) après chaque couche cachée,
        - et d'une activation Sigmoid sur la couche de sortie.
    
        Parameters
        ----------
        layers : list[int]
            Liste des dimensions successives du réseau.
            Exemple : [in_features, hidden1, hidden2, ..., out_features]
    
        Returns
        -------
        nn.Sequential
            Le MLP complet sous forme de séquence PyTorch.
    
        Notes
        -----
        - La couche finale applique systématiquement une Sigmoid, adaptée à des
          sorties dans [0, 1].
        """
        
        layer_list = []
    
        # Ajout des couches cachées : Linear → Normalisation
        # (pour chaque couple consecutive layers[k] → layers[k+1], sauf la dernière)
        for k in range(len(layers) - 2):
            layer_list.extend([
                nn.Linear(layers[k], layers[k+1]),
                self.norm
            ])
    
        # Ajout de la couche finale : Linear → Sigmoid
        layer_list.extend([
            nn.Linear(layers[-2], layers[-1]),
            nn.Sigmoid()
        ])
    
        return nn.Sequential(*layer_list)

    def plot(self, img_array, inputs):
        """
        Affiche côte à côte :
        - l’image originale,
        - l’image prédite par le MLP,
        - l’évolution de la fonction de perte (loss) au cours de l’entraînement.
    
        Parameters
        ----------
        img_array : np.ndarray
            Image originale sous forme de tableau (H, W, 3) utilisée comme référence.
        inputs : array-like or torch.Tensor
            Tableau des coordonnées (ou features) servant d’entrée au réseau.
            Doit correspondre à la grille permettant de reconstruire l’image.
    
        Notes
        -----
        Cette méthode :
        - tensorise les entrées puis les encode avant passage dans le MLP,
        - reshape la sortie du modèle pour retrouver la forme (H, W, 3),
        - trace également la courbe de pertes stockée dans `self.losses`.
        """
        
        # Conversion des inputs en tenseur + récupération du nombre d'échantillons
        inputs, n_samples = tensorise(inputs), inputs.size(0)
        
        # Dimensions de l'image originale
        h, w = img_array.shape[:2]
    
        # Figure principale avec 3 panneaux : original / prédiction / loss
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig)
        
        # --- Image originale ---
        ax_orig = fig.add_subplot(gs[0,0])
        ax_orig.axis('off')
        ax_orig.set_title("Original Image")
        ax_orig.imshow(img_array)
    
        # --- Image prédite ---
        ax = fig.add_subplot(gs[0,1])
        ax.axis('off')
        ax.set_title("Predicted Image")
        # Prédiction → CPU → numpy → reshape en (H, W, 3)
        ax.imshow(
            self.model(self.encoding(inputs))
            .cpu()
            .detach()
            .numpy()
            .reshape(h, w, 3)
        )
    
        # --- Courbe de loss ---
        los = fig.add_subplot(gs[0,2])
        los.set_title("Loss")
    
        # Axe X = 1..N
        los.plot(
            np.linspace(1, len(self.losses), len(self.losses)),
            [loss.item() for loss in self.losses]
        )
        if len(self.losses) ==1:
            lenlosses = 2
        else:
            lenlosses = len(self.losses)
        los.set_xlim(1, lenlosses)
    
        # Évite un ylim min = 0 pile si les pertes sont trop faibles
        maxarray = [0.00000001] + [loss.item() for loss in self.losses]
        los.set_ylim(0, max(maxarray))
    
        # Rafraîchissement non bloquant
        fig.canvas.draw_idle()
        plt.tight_layout()
        plt.ion()
        plt.show()

    def train(self, inputs, outputs, num_epochs=1500, batch_size=1024):
        """
        Entraîne le MLP sur des paires (inputs → outputs) en utilisant un 
        schéma de mini-batchs et l'AMP (Automatic Mixed Precision).
    
        Parameters
        ----------
        inputs : array-like or tensor
            Données d'entrée du réseau, de shape (N, input_dim).
        outputs : array-like or tensor
            Cibles associées, de shape (N, output_dim).
        num_epochs : int, optional
            Nombre total d'époques d'entraînement (default: 1500).
        batch_size : int, optional
            Taille des mini-batchs utilisés à chaque itération (default: 1024).
    
        Notes
        -----
        - Utilise torch.amp.autocast + GradScaler pour un entraînement accéléré en FP16.
        - Les pertes par époque sont stockées dans `self.losses`.
        - Le réseau doit posséder :
            * self.model      : module PyTorch (MLP)
            * self.encoding() : encodage éventuel (Fourier features)
            * self.crit  : fonction de perte
            * self.optim  : optimiseur
        """
    
        # --- Conversion en tensors et récupération du nombre d'échantillons ---
        inputs, outputs, n_samples = tensorise(inputs).to(device), tensorise(outputs).to(device), inputs.size(0)
        self.model = self.model.to(device)
    
        # --- Initialisation du scaler pour l'entraînement en précision mixte ---
        dev = str(device)
        scaler = GradScaler(dev)
    
        # --- Boucle principale sur les époques ---
        for epoch in tqdm(range(num_epochs), desc="train epoch"):
            # Génération d'un ordre aléatoire des indices
            perm = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
    
            # --- Parcours des mini-batchs ---
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
    
                # Fonction interne calculant la perte et les gradients
                def closure():
                    self.optim.zero_grad(set_to_none=True)
                    with autocast(dev): # AMP
                        loss = self.crit(self.model(self.encoding(inputs[idx])),outputs[idx])
                    scaler.scale(loss).backward()
                    return loss
    
                # Calcul de la perte et mise à jour des poids
                loss = closure()
                scaler.step(self.optim)
                scaler.update()
    
                # Accumulation de la perte pour l'époque
                epoch_loss += loss
    
            # --- Stockage de la perte de l'époque ---
            self.losses.append(epoch_loss)

MLP.__init__.help = fPrintDoc(MLP.__init__)
MLP.__repr__.help = fPrintDoc(MLP.__repr__)
MLP.__call__.help = fPrintDoc(MLP.__call__)
MLP.help = fPrintDoc(MLP)
MLP.params.help = fPrintDoc(MLP.params)
MLP.nb_params.help = fPrintDoc(MLP.nb_params)
MLP.neurons.help = fPrintDoc(MLP.neurons)
MLP.Create_MLP.help = fPrintDoc(MLP.Create_MLP)
MLP.plot.help = fPrintDoc(MLP.plot)
MLP.train.help = fPrintDoc(MLP.train)