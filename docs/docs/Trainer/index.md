# **NeuralNetworks.Trainer**

---

`class NeuralNetworks.Trainer (*nets, inputs, outputs=None, init_train_size, final_train_size, optim, init_lr, final_lr, crit, batch_size)` [[source]](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/Trainer/Trainer.py#L16)

---

Classe pour entraîner des réseaux avec mini-batchs et [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

| **Paramètres**     | **Type**                                                                                   | **Optionnel** | 
|--------------------|--------------------------------------------------------------------------------------------|---------------|
| `*nets`            | [`Module`](../module/index.md)                                                             | Non           | 
| `inputs`           | [`torch.Tensor([float])`](https://docs.pytorch.org/docs/stable/tensors.html)               | Non           |
| `outputs`          | [`torch.Tensor([float])`](https://docs.pytorch.org/docs/stable/tensors.html)               | Oui (`None` par défaut) | 
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

## Modes d'apprentissage

Le même objet `Trainer` supporte trois régimes, selon la valeur de `outputs` et du critère :

| **Mode**                  | **`outputs`**                  | **`crit`**                                              |
|---------------------------|--------------------------------|---------------------------------------------------------|
| Supervisé classique       | tenseur de cibles              | `'MSE'`, `'L1'`, ... (voir [`crits`](../Container/crits.md)) |
| Supervisé PINN            | cibles + masque (colonnes extra) | callable custom via [`crits.add`](../Container/add.md) |
| Non supervisé             | `None`                         | callable custom via [`crits.add`](../Container/add.md) |

### Apprentissage non supervisé

Lorsque `outputs=None`, un dummy égal à `inputs` est utilisé en interne (une seule fois au début de [`train`](train.md)) : le critère reçoit ce dummy en troisième argument et peut l'ignorer.

```python
from NeuralNetworks import Trainer, crits

def recon_loss(net, x, _):
    return ((net.train_forward(x) - x) ** 2).mean(dim=0)

crits.add('Recon', recon_loss)

T = Trainer(net, inputs=inputs, crit='Recon')   # outputs=None par défaut
T.train(1000)
```

### PINN (Physics-Informed Neural Network)

Un critère custom peut calculer des dérivées du réseau par rapport à `inputs` via `torch.autograd.grad` (nécessite `inputs.requires_grad_(True)`) :

```python
import torch
from NeuralNetworks import Trainer, crits

def pinn_loss(net, x, y):
    u   = net.train_forward(x)
    du  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    physics = ((-d2u - f(x)) ** 2).mean(dim=0)
    data    = ((u - y) ** 2).mean(dim=0)
    return physics + data

crits.add('PINN', pinn_loss)
inputs.requires_grad_(True)

T = Trainer(net, inputs=inputs, outputs=outputs, crit='PINN')
T.train(1000)
```

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