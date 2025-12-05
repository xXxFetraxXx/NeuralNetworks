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
    init_lr : float, optional
        Taux d’apprentissage initial pour l’optimiseur.
        Default: 1e-3
    Fourier : list[float] or None, optional
            Liste de valeurs sigma pour appliquer plusieurs encodages RFF
            (Random Fourier Features).  
            Si None : aucune transformation, l’entrée est passée via nn.Identity().
            Chaque sigma ajoute un encodage distinct et double la dimension d’entrée.
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
    encodings : list[nn.Module]
        Liste de modules RFF GaussianEncoding ou Identity appliqués aux entrées.
        Un module par valeur sigma dans `Fourier`.
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

    def __init__(self, layers=[1,1,1], init_lr=1e-3, Fourier=None,
                 optim="Adam", crit="MSE", norm="Relu",
                 name="Net", Iscompiled=False):
        """
        Initialise un réseau MLP flexible avec support multi-encodage Fourier,
        choix d’activation, perte, optimiseur, et compilation optionnelle.
    
        Parameters
        ----------
        layers : list[int], optional
            Dimensions successives du réseau (entrée → couches cachées → sortie).
            Le premier élément est utilisé comme input_size avant encodage Fourier.
            Default: [1, 1, 1]
        init_lr : float, optional
            Taux d’apprentissage initial pour l’optimiseur.
            Default: 1e-3
        Fourier : list[float] or None, optional
            Liste de valeurs sigma pour appliquer plusieurs encodages RFF
            (Random Fourier Features).  
            Si None : aucune transformation, l’entrée est passée via nn.Identity().
            Chaque sigma ajoute un encodage distinct et double la dimension d’entrée.
        optim : str, optional
            Nom de l’optimiseur (doit appartenir à `optim_list(self, init_lr)`).
            Default: "Adam"
        crit : str, optional
            Nom de la fonction de perte (doit appartenir à `crit_list`).
            Default: "MSE"
        norm : str, optional
            Nom de la fonction d’activation entre couches cachées
            (doit appartenir à `norm_list`).
            Default: "Relu"
        name : str, optional
            Nom du modèle, utilisé pour l’identification.
            Default: "Net"
        Iscompiled : bool, optional
            Si True, compile le MLP avec `torch.compile` pour accélérer l’inférence.
            Si GCC est absent, la compilation est automatiquement désactivée.
            Default: False
    
        Attributes
        ----------
        losses : list
            Historique des pertes pendant l’entraînement.
        layers : list[int]
            Dimensions du réseau, modifiées si un encodage Fourier est appliqué
            (l’entrée devient 2 * encoded_size).
        encodings : list[nn.Module]
            Liste de modules RFF GaussianEncoding ou Identity appliqués aux entrées.
            Un module par valeur sigma dans `Fourier`.
        norm : nn.Module
            Fonction d’activation utilisée dans les couches cachées.
        crit : nn.Module
            Fonction de perte PyTorch.
        model : nn.Sequential
            Réseau MLP construit dynamiquement via `Create_MLP()`.
        optim : torch.optim.Optimizer
            Optimiseur associé au MLP.
        name : str
            Nom du réseau.
        f : nn.Linear
            Couche linéaire appliquée après le MLP, prenant en entrée
            (nb_encodings * layers[-1]) et renvoyant layers[-1].
            Utilisée pour agréger les sorties des différents encodages.
        """
    
        super().__init__()
    
        # --- Initialisation des attributs de base ---
        self.losses, self.layers, self.Fourier = [], layers.copy(), Fourier
        self.name = name
    
        # --- Encodage Fourier (RFF) ou passthrough ---
        self.encodings = []
        if self.Fourier is None:
            self.encodings.append(nn.Identity().to(device))  # passthrough si pas de Fourier
        else:
            for sigma_val in Fourier:
                self.encodings.append(rff.layers.GaussianEncoding(
                    sigma=sigma_val,
                    input_size=self.layers[0],
                    encoded_size=self.layers[1]
                ).to(device))
            self.layers[0] = self.layers[1] * 2  # chaque entrée est doublée après encodage
        
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
        self.optim = optim_list(self, init_lr).get(optim)
        if self.optim is None:
            print("")
            print (f"{optim} n'est pas reconnu")
            self.optim = optim_list(self, init_lr).get("Adam")
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
        self.f = nn.Linear(len(self.encodings) * self.layers[-1], self.layers[-1]).to(device)

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
        ); plt.show(); return ""

    def __call__(self, x):
        """
        Effectue une inférence complète en appliquant :
        - chaque encodage d'entrée défini dans `self.encodings`,
        - un passage dans le MLP (`self.model`) pour chaque encodage,
        - une concaténation des sorties,
        - une couche linéaire finale (`self.f`) pour produire la prédiction.
    
        Cette méthode permet d’utiliser l’objet comme une fonction :
            y = net(x)
    
        Parameters
        ----------
        x : array-like
            Entrée(s) à prédire. Peut être un tableau NumPy, une liste Python
            ou un tenseur PyTorch convertible par `tensorise()`.
    
        Returns
        -------
        np.ndarray
            Sortie finale du modèle après :
            encodage(s) → MLP → concaténation → couche finale.
            Résultat renvoyé sous forme de tableau NumPy CPU aplati.
        """
        
        # Inférence sans calcul de gradient (plus rapide et évite la construction du graphe)
        with torch.no_grad():
            inputs = tensorise(x)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0) 
            
            # --- Initialisation du scaler pour l'entraînement en précision mixte ---

            results_list = []
            for encoding in self.encodings:
                results_list.append(self.model(encoding(inputs)))
            return self.f(torch.cat(results_list, dim=1)).cpu().numpy().flatten()

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
    
        # Ajout de la couche finale : Linear
        layer_list.extend([
            nn.Linear(layers[-2], layers[-1])
        ])
    
        return nn.Sequential(*layer_list)

    def train(self, inputs, outputs, num_epochs=1500, batch_size=1024):
        """
        Entraîne le modèle en utilisant un schéma mini-batch et la précision
        mixte (AMP). Chaque batch subit :
    
            encodages multiples → MLP → concaténation → couche finale → perte
    
        Ce training met également à jour dynamiquement le learning rate.
    
        Parameters
        ----------
        inputs : array-like or torch.Tensor
            Données d'entrée, shape (N, input_dim). Sont converties en
            tenseur PyTorch et envoyées sur `device`.
        outputs : array-like or torch.Tensor
            Cibles correspondantes, shape (N, output_dim).
        num_epochs : int, optional
            Nombre total d'époques d'entraînement.
            Default: 1500.
        batch_size : int, optional
            Taille des mini-batchs. Default: 1024.
    
        Notes
        -----
        - Utilise `torch.amp.autocast` et `GradScaler` pour un entraînement
          accéléré et stable en précision mixte (FP16/FP32).
        - Pour chaque mini-batch :
            * chaque module dans `self.encodings` est appliqué aux entrées,
            * chaque encodage passe dans `self.model`,
            * les sorties sont concaténées,
            * la couche finale `self.f` produit la prédiction,
            * la perte est évaluée via `self.crit`.
        - Le learning rate est ajusté après chaque époque.
        - Les pertes d'époques sont enregistrées dans `self.losses`.
        """
    
        # --- Conversion en tensors et récupération du nombre d'échantillons ---
        inputs, outputs = tensorise(inputs).to(device), tensorise(outputs).to(device)
        self.model = self.model.to(device)
        n_samples = inputs.size(0)

        # --- Initialisation du scaler pour l'entraînement en précision mixte ---
        dev = str(device)
        scaler = GradScaler(dev)

        def update_lr(optimizer, loss):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.005*np.where(loss <=0, 0,
                np.where(loss >=1, 1,
                np.sqrt(loss)/(2 - loss**2)))

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
                        results_list = []
                        for encoding in self.encodings:
                            results_list.append(self.model(encoding(inputs[idx])))
                        loss = self.crit(self.f(torch.cat(results_list, dim=1)),outputs[idx])
                    scaler.scale(loss).backward()
                    return loss

                # Calcul de la perte et mise à jour des poids
                loss = closure()
                scaler.step(self.optim)
                scaler.update()

                # Accumulation de la perte pour l'époque
                epoch_loss += loss.item()

            # --- Stockage de la perte de l'époque ---
            self.losses.append(epoch_loss)
            update_lr(self.optim,self.losses[-1])

def losses(*nets):
    """
    Affiche les courbes de pertes (training loss) de plusieurs réseaux MLP.

    Parameters
    ----------
    nets : MLP
        Un ou plusieurs réseaux possédant un attribut `.losses`
        contenant l'historique des pertes (liste de float).

    Notes
    -----
    - L’axe X correspond aux itérations (epochs ou steps).
    - L’axe Y correspond à la valeur de la perte.
    - La fonction utilise matplotlib en mode interactif pour affichage dynamique.
    """

    # --- Initialisation de la figure ---
    fig = plt.figure(figsize=(5, 5))

    # --- Définition des limites des axes ---
    all_losses = [[loss for loss in net.losses] for net in nets]
    if max(len(lst) for lst in all_losses) == 1:
        lenlosses = 2
    else:
        lenlosses = max(len(lst) for lst in all_losses)
    plt.xlim(1, lenlosses)

    # --- Tracé des courbes de pertes pour chaque réseau ---
    for k, net in enumerate(nets):
        steps = np.linspace(1, len(net.losses), len(net.losses))  # epochs
        plt.plot(np.arange(1, len(all_losses[k])+1), all_losses[k],label = net.name)
    plt.yscale('log', nonpositive='mask')
    # --- Affichage ---
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Résidus")
    fig.canvas.draw_idle()
    plt.tight_layout()
    plt.ion()  # mode interactif
    plt.show()

losses.help = fPrintDoc(losses)

MLP.__init__.help = fPrintDoc(MLP.__init__)
MLP.__repr__.help = fPrintDoc(MLP.__repr__)
MLP.__call__.help = fPrintDoc(MLP.__call__)
MLP.help = fPrintDoc(MLP)
MLP.params.help = fPrintDoc(MLP.params)
MLP.nb_params.help = fPrintDoc(MLP.nb_params)
MLP.neurons.help = fPrintDoc(MLP.neurons)
MLP.train.help = fPrintDoc(MLP.train)