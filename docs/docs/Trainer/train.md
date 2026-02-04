# **NeuralNetworks.Trainer.train**

---

`NeuralNetworks.Trainer.train (num_epochs, disable_tqdm, benchmark)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/Trainer/Trainer.py#L44)

---

Lancement d'un entrainement avec le trainer définit.

| **Paramètres** | **Type**                                                                                 | **Optionnel** |
|----------------|------------------------------------------------------------------------------------------|---------------|
| `num_epochs`   | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Oui           |
| `disable_tqdm` | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool)           | Oui           |
| `benchmark`    | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool)           | Oui           |

---

???+ example "Lancement d'un entrainement"

    ```python title="train_exemple.py" linenums="1"
    from NeuralNetworks import Trainer

    T = Trainer (
        net              , # (1)!
        inputs  = inputs , # (2)!
        outputs = outputs, # (3)!
    )

    T.train (
        num_epochs   = 1000 ,
        disable_tqdm = False,
        benchmark    = False
    )
    ```

    1. Voir [`Module`](../module/index.md)
    2. La définition de inputs n'est pas explicitée ici
    3. La définition de outputs n'est pas explicitée ici