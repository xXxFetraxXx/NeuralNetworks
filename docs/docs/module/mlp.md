# **NeuralNetworks.MLP**

---

`class MLP (input_size, output_size, hidden_layers, sigmas, fourier_input_size, nb_fourier, norm, name)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/MLP/MLP.py#L11)

---

inhérite des propriétés de la classe [`Module`](index.md)   

Cette classe fournit :

- Un [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) entièrement configurable (dimensions, activation).
- Option d'encodage [Fourier Features](https://en.wikipedia.org/wiki/Random_feature#Random_Fourier_feature) sur les entrées.

---

Permet de construire rapidement un réseau de neurones multicouches rapidement sans connaisances profondes de pytorch.

| **Paramètres**       | **Type**                                                                                         | **Optionnel** | **Default value** |
| :------------------- | :----------------------------------------------------------------------------------------------- | :------------ | :---------------- |
| `input_size`         | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `1`               |
| `output_size`        | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `1`               |
| `hidden_layers`      | [`list[int]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)   | Oui           | `[1]`             |
| `sigmas`             | [`list[float]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) | Oui           | `None`            |
| `fourier_input_size` | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `2`               |
| `nb_fourier`         | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `8`               |
| `norm`               | [`norm`](../Container/norms.md)                                                                  | Oui           | `'Relu'`          |
| `name`               | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  | Oui           | `'Net'`           |

---

???+ example

    ```py title="MLP_exemple.py" linenums="1"
    from NeuralNetworks import MLP

    net = MLP(
        input_size    = 2,
        hidden_layers = [512,512,512,512,512,512,512,512,512,512],
        output_size   = 3,
        nb_fourier    = 256,
        sigmas        = [0.1,1],
        norm          = "Relu",
        name          = "Net"
    )
    ```