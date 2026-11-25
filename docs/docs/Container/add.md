# **NeuralNetworks.Container.add**

---

`NeuralNetworks.Container.add (name, value)` [[source]](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/_Dependances/pytorch.py#L80)

---

Ajoute l'entrée `name` au dictionnaire du conteneur. Permet d'enregistrer de nouvelles valeurs dans `norms`, `crits` ou `optims` à l'exécution.

| **Paramètres** | **Type** | **Optionnel** |
|----------------|----------|---------------|
| `name`         | [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str) | Non |
| `value`        | `any`    | Non           |

---

???+ example "Ajouter un critère personnalisé à `crits`"

    ```python title="crit_custom.py" linenums="1"
    import torch.nn as nn
    from NeuralNetworks import crits
    from NeuralNetworks._Dependances.pytorch import StandardCriterion

    crits.add('MonHuber', StandardCriterion(nn.HuberLoss(reduction='none', delta=2.0)))

    # Utilisable ensuite comme n'importe quel critère :
    # Trainer(net, inputs=inputs, outputs=outputs, crit='MonHuber')
    ```

???+ example "Ajouter un critère entièrement custom (PINN, non supervisé, ...)"

    ```python title="crit_pinn.py" linenums="1"
    from NeuralNetworks import crits

    def pinn_loss(net, x, y):
        u   = net.train_forward(x)
        du  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        physics = ((-d2u - f(x)) ** 2).mean(dim=0)
        data    = ((u - y) ** 2).mean(dim=0)
        return physics + data

    crits.add('PINN', pinn_loss)
    ```

    La signature d'un critère est `(net, inputs, outputs) -> torch.Tensor`. Le critère reçoit toujours trois arguments ; en apprentissage non supervisé (`outputs=None`), il reçoit un dummy et peut l'ignorer.
