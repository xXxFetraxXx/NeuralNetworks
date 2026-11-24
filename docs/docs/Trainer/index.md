# **NeuralNetworks.Trainer**

---

`class NeuralNetworks.Trainer (*nets, inputs, outputs, init_train_size, final_train_size, optim, init_lr, final_lr, crit, batch_size)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/Trainer/Trainer.py#L16)

---

Classe pour entraîner des réseaux avec mini-batchs et [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

| **Paramètres**     | **Type**                                                                                   | **Optionnel** | 
|--------------------|--------------------------------------------------------------------------------------------|---------------|
| `*nets`            | [`Module`](../module/index.md)                                                             | Non           | 
| `inputs`           | [`torch.Tensor([float])`](https://docs.pytorch.org/docs/stable/tensors.html)               | Non           |
| `outputs`          | [`torch.Tensor([float])`](https://docs.pytorch.org/docs/stable/tensors.html)               | Non           | 
| `init_train_size`  | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           |
| `final_train_size` | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           |
| `optim`            | [`optim`](../Container/optims.md)                                                          | Oui           |
| `init_lr`          | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           |
| `final_lr`         | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           | 
| `crit`             | [`crit`](../Container/crits.md)                                                            | Oui           |
| `batch_size`       | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)   | Oui           |

---

## Taille d'entraînement progressive

`init_train_size` et `final_train_size` sont des fractions du dataset (entre `0` et `1`). Le nombre d'échantillons utilisés croît **linéairement** de `init_train_size × N` à `final_train_size × N` sur les époques (arrondi à l'entier supérieur), où `N` est le nombre total d'échantillons.

À chaque époque, les échantillons sont tirés par [**Latin Hypercube Sampling**](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) 1D : l'intervalle `[0, N)` est divisé en strates égales et un indice aléatoire est tiré dans chaque strate, assurant une couverture uniforme de l'ensemble des données.

---

???+ example "Initialisation d'un trainer"

    ```python title="Trainer_exemple.py" linenums="1"
    from NeuralNetworks import Trainer

    T = Trainer (
        net                       , # (1)!
        inputs           = inputs , # (2)!
        outputs          = outputs, # (3)!
        init_train_size  = 0.001  , 
        final_train_size = 1      , 
        init_lr          = 1e-3   , 
        final_lr         = 1e-5   , 
        optim            = 'Adam' , 
        crit             = 'MSE'  , 
        batch_size       = 163840   
    )
    ```

    1. Voir [`Module`](../module/index.md)
    2. La définition de inputs n'est pas explicitée ici
    3. La définition de outputs n'est pas explicitée ici