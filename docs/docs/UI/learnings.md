# **NeuralNetworks.learnings**

---

`NeuralNetworks.learnings (*nets, fig_size, color)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/_UI/Learnings.py#L11)

---

Affiche les taux d'apprentissage en fonction des époques d'entrainement des réseaux.

| **Paramètres** | **Type**                                                                                 | **Optionnel** |
|----------------|------------------------------------------------------------------------------------------|---------------|
| `*nets`        | [`Module`](../module/index.md)                                                           | Non           | 
| `fig_size`     | [`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | oui           |
| `color`        | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)          | oui           |

---

???+ example

    ```py title="learnings.py" linenums="1"
    from NeuralNetworks import MLP, Trainer, learnings

    hidden_layers = [256,256,256,256,256,256,256,256,256,256]
    net = MLP(2, hidden_layers, 3)

    T = Trainer (net, inputs = inputs, outputs = outputs) # (1)!
    T.train(1500)

    learnings (net, fig_size = 10)
    ```

    1. La définition de inputs et outputs n'est pas explicitée ici

    ![Learnings](Learnings_alt.png#only-light)
    ![Learnings](Learnings.png#only-dark)