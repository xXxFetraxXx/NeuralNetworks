# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import nn, torch, device, np, onnx, simplify

class Module (nn.Module):
    """
    Classe de base pour les modèles.
    Gère le forward entraînement, l'inférence et le suivi des métriques.
    """

    def __init__ (self,
        _name : str = "Net",
        **Reconstruction_data): # Nom du modèle
        """
        Initialise le module.
        """
        super ().__init__ ()  
        self.losses              = []                  # Tracker des résidus
        self.learnings           = []                  # Tracker des lrs
        self.name                = _name               # Nom du module
        self.Reconstruction_data = Reconstruction_data # Données de reconstruction
        

    def _forward (self, x : torch.Tensor):
        """
        Forward interne à implémenter dans les classes filles.

        Utile pour VAEs :
        Si la sortie d'entrainement est différente de la sortie d'utilisation,
        alors redéfinir train_forward pour l'entrainement.
        """
        raise Exception ("_forward n'est pas défini dans la classe")

    def train_forward (self, x : torch.Tensor): # Forward de train (avec gradients)
        return self._forward (x)

    def forward(self, x: torch.Tensor):         # Forward d'inférence (sans gradients)
        device = next (self.parameters ()).device
    
        with torch.no_grad ():
            x = x.unsqueeze (0) if x.dim () == 1 else x
            x = x.to (device)
            output = self._forward (x)
            return output.cpu ().numpy ().flatten ()

    @property
    def nb_params (self):
        """
        Affiche le nombre total de paramètres et ceux entraînables.
        """
        total     = sum (p.numel () for p in self.parameters ())
        trainable = sum (p.numel () for p in self.parameters () if p.requires_grad)
        print (f"Nombre total de paramètres : {total}")
        print (f"Nombre de paramètres entraînables : {trainable}")

    @property
    def save(self):
        state_dict = {
            k: v.detach().cpu().numpy()
            for k, v in self.state_dict().items()
        }
        np.savez(f"{self.name}.npz", allow_pickle = False, # Initialise l'enregistement
            **state_dict                                 , # Données d'état
            losses    = np.asarray(self.losses)          , # Données de loss
            learnings = np.asarray(self.learnings)       , # Données de lr
            **self.Reconstruction_data                   ) # Données de reconstruction

    @classmethod
    def load(cls, path, device="cpu"):
        data = np.load(path, allow_pickle = False) # Lis les données
        index = list(data.keys()).index('losses')  # Indice de séparation
        
        obj = cls (**{k: data[k] for k in list(data.keys())[index + 2:]})

        state_dict = {
            k: torch.from_numpy(data[k]).to(device)
            for k in obj.state_dict().keys()
        }
        
        obj.losses    = data ["losses"].tolist( )    # Charge les données de loss
        obj.learnings = data ["learnings"].tolist( ) # Charge les données de lr
        obj.load_state_dict  (state_dict)            # Charge le réseau

        return obj

    @property
    def _dummy_input(self):
        raise Exception ("_dummy_input n'est pas défini dans la classe")

    @property
    def onnx_save (self):
        print ("Sauvegarde en format .onnx")
        torch.onnx.export (self,
            self._dummy_input,                           # Renseigne la input shape
            f"{self.name}.onnx",                         # Nom du fichier de sauvegarde
            opset_version  = 18,                         #
            dynamo         = True,                       #
            input_names    = ["inputs"],                 # Nom des inputs
            output_names   = ["outputs"],                # Nom des outputs
            dynamic_shapes = {                           
                "x": {0: torch.export.Dim("batch_size")} #
            }
        )

        print ("Simplification du .onnx")
        onnx_model_simp, check = simplify (onnx.load (f"{self.name}.onnx"))
        assert check

        onnx.save (onnx_model_simp, f"{self.name}_simplified.onnx")
        print ("Fin de l'enregistrement")