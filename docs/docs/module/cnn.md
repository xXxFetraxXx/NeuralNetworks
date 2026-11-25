# **NeuralNetworks.CNN**

---

`class CNN (in_channels, output_size, conv_layers, kernel_size, norm, name)` [[source]](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/CNN/CNN.py#L12)

---

inhérite des propriétés de la classe [`Module`](index.md)   

Cette classe fournit :

- Un [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) entièrement configurable pour traiter des images.
- Une pile de blocs `Conv2d → activation → MaxPool2d`, suivie d'une tête de classification `AdaptiveAvgPool2d → Flatten → Linear`.

---

L'entrée est un tenseur 4D `[batch, in_channels, H, W]` et la sortie est `[batch, output_size]`. Grâce au `AdaptiveAvgPool2d((1, 1))` final, **n'importe quelle taille d'image H×W est acceptée**.

| **Paramètres**  | **Type**                                                                                         | **Optionnel** | **Default value** |
| :-------------- | :----------------------------------------------------------------------------------------------- | :------------ | :---------------- |
| `in_channels`   | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `1`               |
| `output_size`   | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `1`               |
| `conv_layers`   | [`list[int]`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)   | Oui           | `[1]`             |
| `kernel_size`   | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)         | Oui           | `3`               |
| `norm`          | [`norm`](../Container/norms.md)                                                                  | Oui           | `'Relu'`          |
| `name`          | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)                  | Oui           | `'Net'`           |

---

???+ example

    ```py title="CNN_exemple.py" linenums="1"
    from NeuralNetworks import CNN

    net = CNN(
        in_channels  = 3,
        output_size  = 10,
        conv_layers  = [32, 64, 128],
        kernel_size  = 3,
        norm         = "Relu",
        name         = "MonCNN"
    )
    ```
