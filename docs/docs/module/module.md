# NeuralNetworks.Module

`NeuralNetworks.Module (_name, **Reconstruction_data)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/_shared/module.py#L10)

Module complet pour faciliter la création et l'entraînement de modèles [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) en fournissant des classes préconfigurées.

| **sous-classes** | **Description**                                                                                         |
| :--------------- | :------------------------------------------------------------------------------------------------------ |
| [`MLP`](mlp.md)  | Classe préfabriquée de [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)     |
| [`VAE`](vae.md)  | Classe préfabriquée de [Variational Autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) |

---

## **Attributs**

| **Attributs**   | **Type**                                                                                         | **Description**                                                                 |
| :-------------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| `MLP.losses`    | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Historique des pertes cumulées lors de l'entraînement                           |
| `MLP.learnings` | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Historique des taux d'apprentissage utilisées lors de l'entraînement            |
| `MLP.model`     | [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)       | MLP complet construit dynamiquement                                             |
| `MLP.name`      | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  | Nom du réseau                                                                   |
| `MLP.save`      | [`method`](https://docs.python.org/3/library/stdtypes.html#methods)                              | Enregistre le réseau dans un format propriétaire sécurisé                       |
| `MLP.load`      | [`method`](https://docs.python.org/3/library/stdtypes.html#methods)                              | Charge un réseau du format propriétaire sécurisé, peut continuer l'entrainement | 
| `MLP.onnx_save` | [`method`](https://docs.python.org/3/library/stdtypes.html#methods)                              | Enregistre le réseau dans le format [`onnx`](https://onnx.ai)                   |