# **NeuralNetworks.losses**

`NeuralNetworks.losses (*nets, fuse_losses, names, fig_size, color)` [[source]](https://github.com/xXxFetraxXx/NeuralNetworks/blob/main/src/NeuralNetworks/_UI/Losses.py#L11)

Affiche les résidus en fonction des époques d'entrainement des réseaux.

| **Paramètres** | **Type**                                                                       | **Optionnel** | **Description**                                                          |
|----------------|--------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------|
| `*nets`        | [`Module`](../module/module.md)                                                            | Non           | Réseaux pour lesquels afficher les résidus.                              |
| `fuse_losses`  | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) | oui           | Fusionner les résidus des sorties de chaques réseaux en un résidu moyen. |
| `names`        | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) | oui           | Noms des sorties des réseaux.                                            |
| `fig_size`     | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) | oui           | Taille de la figure.                                                     |
| `color`        | [`boolean`](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) | oui           | Couleur des axes.                                                        |


???+ example

    === "Sans fusion des résidus"

        ```py title="losses_exemple.py" linenums="1"
        from NeuralNetworks import MLP, Trainer, learnings

        hidden_layers = [256,256,256,256,256,256,256,256,256,256]
        net = MLP(2, hidden_layers, 3)

        T = Trainer (net, inputs = inputs, outputs = outputs) # (1)!
        T.train(1500)

        losses (net, fuse_losses = False, names = ["Sortie 1", "Sortie 2", "Sortie 3"], fig_size = 10)
        ```

        1. La définition de inputs et outputs n'est pas explicitée ici

        ![Learnings](Losses_alt.png#only-light)
        ![Learnings](Losses.png#only-dark)

    === "Avec fusion des résidus"

        ```py title="fused_losses_exemple.py" linenums="1"
        from NeuralNetworks import MLP, Trainer, learnings

        hidden_layers = [256,256,256,256,256,256,256,256,256,256]
        net = MLP(2, hidden_layers, 3)

        T = Trainer (net, inputs = inputs, outputs = outputs) # (1)!
        T.train(1500)

        losses (net, fuse_losses = True, fig_size = 10)
        ```

         1. La définition de inputs et outputs n'est pas explicitée ici

        ![Learnings](LossesGroupped_alt.png#only-light)
        ![Learnings](LossesGroupped.png#only-dark)