# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import torch, nn, np

class FourierEncoding (nn.Module):
    """
    Encodage de Fourier aléatoire pour enrichir la représentation des entrées.

    Cette couche projette les entrées dans un espace fréquentiel à l'aide
    d'une matrice de projection apprise, puis applique des fonctions sinus
    et cosinus afin de capturer des variations à haute fréquence.

    Parameters
    ----------
    nb_fourier : int
        Nombre de composantes de Fourier.
    input_size : int
        Dimension des entrées.
    sigma : float
        Écart-type utilisé pour l'initialisation de la matrice de projection.
    """
    def __init__ (self, nb_fourier, input_size, sigma):
        super ().__init__ ()
        self.B = nn.Parameter (torch.randn (nb_fourier, input_size) * sigma)

    def forward (self, x):
        """
        Applique l'encodage de Fourier aux entrées.

        Parameters
        ----------
        x : torch.Tensor
            Tensor d'entrée de shape `(N, input_size)`.

        Returns
        -------
        torch.Tensor
            Tensor encodé de shape `(N, 2 * nb_fourier)`, correspondant
            à la concaténation des cosinus et sinus.
        """
        vp = 2 * np.pi * x @ self.B.T
        return torch.cat ((torch.cos (vp), torch.sin (vp)), dim = -1)

def encode (input_size, output_size, sigmas, fourier_input_size, nb_fourier):
    """
    Construit les modules d'encodage (Fourier ou identité) et la couche de fusion associée.

    Si `sigmas` est `None`, aucun encodage de Fourier n'est appliqué et les
    entrées sont transmises directement au réseau.
    Sinon, plusieurs encodages de Fourier sont créés (un par sigma), et
    leurs sorties sont fusionnées via une couche linéaire.

    Parameters
    ----------
    input_size : int
        Dimension des entrées.
    output_size : int
        Dimension de sortie du réseau.
    sigmas : list[float] ou None
        Liste des paramètres sigma pour les encodages de Fourier.
    fourier_input_size : int
        Dimension attendue après encodage (non utilisée directement ici,
        mais conservée pour cohérence avec l'architecture globale).
    nb_fourier : int
        Nombre de composantes de Fourier par encodage.

    Returns
    -------
    encodings : torch.nn.ModuleList (scripté)
        Liste des modules d'encodage (Fourier ou identité).
    f : torch.nn.Module (scripté)
        Module de fusion des encodages (identité ou couche linéaire).
    """
    if sigmas is None:
        encodings = nn.ModuleList (
            [nn.Identity ()]
        )
        f = nn.Identity ()
    else:
        encodings = nn.ModuleList (
            [FourierEncoding (nb_fourier, input_size, sigma) for sigma in sigmas]
        )
        f = nn.Linear (len (encodings) * output_size, output_size)
    return torch.jit.script (encodings), torch.jit.script (f)