# **NeuralNetworks.VAE**

---

`class VAE (imsize, latentsize, labelsize, channels, linear_channels, name, norm, norm_cc)` [[Source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/VAE/VAE.py#L11)

---

inhérite des propriétés de la classe [`Module`](index.md)   

Cette classe fournit :

- Un [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder) entièrement configurable.

---

| **Paramètres**    | **Type**                                                                                       | **Optionnel** |
|-------------------|------------------------------------------------------------------------------------------------|---------------|
| `imsize`          | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)       | Non           |
| `latentsize`      | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)       | Non           |
| `labelsize`       | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)       | Non           |
| `channels`        | [`list[int]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Oui           |
| `linear_channels` | [`list[int]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Oui           |
| `name`            | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                | Oui           |
| `norm`            | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                | Oui           |
| `norm_cc`         | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                | Oui           |

???+ example

    ```py title="VAE_exemple.py" linenums="1"
    from NeuralNetworks import VAE

    # WIP

    ```