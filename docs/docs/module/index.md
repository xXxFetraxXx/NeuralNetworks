# **NeuralNetworks.Module**

---

`class NeuralNetworks.Module (_name, **Reconstruction_data)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/_shared/module.py#L10)

---

Module complet pour faciliter la création et l'entraînement de modèles [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) en fournissant des classes préconfigurées.

| **sous-classes** | **Description**                                                                                         |
| :--------------- | :------------------------------------------------------------------------------------------------------ |
| [`MLP`](mlp.md)  | Classe préfabriquée de [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)     |
| [`VAE`](vae.md)  | Classe préfabriquée de [Variational Autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) |

---

| **Attributs**   | **Type**                                                                                         |
| :-------------- | :----------------------------------------------------------------------------------------------- |
| `MLP.losses`    | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) |
| `MLP.learnings` | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) |
| `MLP.model`     | [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)       |
| `MLP.name`      | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  |