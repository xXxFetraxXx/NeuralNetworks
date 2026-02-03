# **NeuralNetworks.Trainer**

`NeuralNetworks.Trainer (*nets, inputs, outputs, init_train_size, final_train_size, optim, init_lr, final_lr, crit, batch_size)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/Trainer/Trainer.py#L15)

Classe pour entraîner des réseaux avec mini-batchs et [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

| **Paramètres**     | **Type**                                                                                        | **Optionnel** | **Description**                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------|
| `*nets`            | [`Module`](../module/module.md)                                                                                   | Non           | Réseaux pour lesquels le trainer va entrainer.                                                                  |
| `inputs`           | [`torch.Tensor([float])`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) | Non           | Données en entrée au réseau.                                                                                    |
| `outputs`          | [`torch.Tensor([float])`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) | Non           | Données en sortie au réseau.                                                                                    |
| `init_train_size`  | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Fraction du dataset initiale.  Default: `0.01`                                                                  |
| `final_train_size` | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Fraction du dataset finale.  Default: `1`                                                                       |
| `optim`            | [`optim`](../Containers/optims.md)                                                                              | Oui           | Nom de l’optimiseur à utiliser (doit exister dans `optims()`). Default: `'Adam'`                                |
| `init_lr`          | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Taux d’apprentissage initial pour l’optimiseur. Default: `1e-3`                                                 |
| `final_lr`         | [`float`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)      | Oui           | Taux d’apprentissage final pour l’optimiseur. Default: `1e-5`                                                   |
| `crit`             | [`crit`](../Containers/crits.md)                                                                                | Oui           | Fonction de perte à utiliser (doit exister dans `crits()`). Default: `'MSE'`                                    |
| `batch_size`       | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)        | Oui           | Taille des minibatchs. Default: `1024`                                                                          |

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

    1. Voir [`Module`](../module/module.md)
    2. La définition de inputs n'est pas explicitée ici
    3. La définition de outputs n'est pas explicitée ici