# Exemples

Des notebooks d'exemple sont fournis dans [`../../../Exemples`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/tree/main/src/NeuralNetworks/Exemples) :

| **Notebook**                          | **Contenu**                                                                                          |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------- |
| [`MLP.ipynb`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Exemples/MLP.ipynb)             | Construction d'un `MLP`, forward, sauvegarde / chargement `.npz`                                    |
| [`CNN.ipynb`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Exemples/CNN.ipynb)             | Construction d'un `CNN`, forward sur images 4D, classification avec `CrossEntropy`                  |
| [`Trainer.ipynb`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Exemples/Trainer.ipynb)     | Entraînement supervisé complet (`y = sin(2πx)`), courbes `losses` / `learnings`                     |
| [`PINN.ipynb`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Exemples/PINN.ipynb)           | PINN : résolution d'une EDO par critère physique custom (dérivées automatiques)                     |
| [`Unsupervised.ipynb`](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Exemples/Unsupervised.ipynb) | Non supervisé avec critère custom (projection sur le cercle unité), et autoencodeur en remarque   |

---

## Supervisé

```py title="Extrait de Trainer.ipynb"
from NeuralNetworks import MLP, Trainer, losses, learnings

trainer = Trainer(net,
                  inputs=inputs, outputs=outputs,
                  init_train_size=1.0, final_train_size=1.0,
                  crit="MSE", optim="Adam",
                  init_lr=1e-3, final_lr=1e-5,
                  batch_size=64)
trainer.train(200)

losses(net)
learnings(net)
```

## PINN (équations physiques)

```py title="Extrait de PINN.ipynb"
def pinn_crit(net, xb, _):
    xb = xb.detach().requires_grad_(True)
    u = net.train_forward(xb)
    dudx = torch.autograd.grad(u.sum(), xb, create_graph=True)[0]
    d2u = torch.autograd.grad(dudx.sum(), xb, create_graph=True)[0]
    phys = (d2u + torch.pi ** 2 * u).pow(2).mean()   # résidu de l'EDO
    bc = u[0].pow(2) + (dudx[0] - torch.pi).pow(2)   # conditions aux limites
    return phys + 100 * bc
```

## Non supervisé (critère custom)

```py title="Extrait de Unsupervised.ipynb"
def cercle_crit(net, xb, _):
    y = net.train_forward(xb)
    return ((y.pow(2).sum(dim=1) - 1) ** 2).mean()   # distance au cercle unité

trainer = Trainer(net, inputs=X, crit=cercle_crit, ...)   # pas d'outputs
```

> Sans `outputs` et avec un critère prédéfini (`"MSE"`), le `Trainer` retombe sur un **autoencodeur** : `loss = MSE(net(X), X)`.
