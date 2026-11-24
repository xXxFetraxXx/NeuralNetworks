# **NeuralNetworks.Module**

---

`class NeuralNetworks.Module (_name, **Reconstruction_data)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/_shared/module.py#L11)

---

Module complet pour faciliter la création et l'entraînement de modèles [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) en fournissant des classes préconfigurées.

| **sous-classes** | **Description**                                                                                     |
| :--------------- | :-------------------------------------------------------------------------------------------------- |
| [`MLP`](mlp.md)  | Classe préfabriquée de [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) |

---

| **Attributs**         | **Type**                                                                                         |
| :-------------------- | :----------------------------------------------------------------------------------------------- |
| `Module.losses`       | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) |
| `Module.learnings`    | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) |
| `Module.model`        | [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)       |
| `Module.name`         | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  |
| `Module.nb_params`    | [`property`](https://docs.python.org/3/library/functions.html#property) — affiche le nombre de paramètres |