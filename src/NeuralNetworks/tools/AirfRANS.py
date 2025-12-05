# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import *
from airfrans import *

def download(path,unzip = True, OpenFOAM = False):
    """
    Télécharge le dataset AirfRANS dans le dossier spécifié.

    Cette fonction est un simple wrapper autour :
        dataset.download(root=path, file_name='AirfRANS', unzip=True, OpenFOAM=True)

    Les arguments `unzip` et `OpenFOAM` sont actuellement ignorés par la fonction
    et forcés à True dans l’appel interne.

    Parameters
    ----------
    path : str
        Chemin du dossier dans lequel le dataset doit être téléchargé.
    unzip : bool, optional
        Paramètre non utilisé. Le téléchargement interne force `unzip=True`.
    OpenFOAM : bool, optional
        Paramètre non utilisé. Le téléchargement interne force `OpenFOAM=True`.

    Notes
    -----
    - Le fichier téléchargé s’appelle `'AirfRANS'`.
    - Le dataset est automatiquement décompressé.
    - Le format OpenFOAM est toujours inclus.
    """
    dataset.download(root = path, file_name = 'AirfRANS', unzip = True, OpenFOAM = True)