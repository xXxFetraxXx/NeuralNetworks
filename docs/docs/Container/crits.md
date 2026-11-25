# **NeuralNetworks.crits** 

---

`NeuralNetworks.crits` [[source]](https://gitlab.isae-supaero.fr/a.brun/NeuralNetworks/-/blob/main/src/NeuralNetworks/_Dependances/pytorch.py#L123)

---

inhérite des propriétés de la classe [`Container`](index.md)   

| **Valeurs**              | **Module PyTorch**                                                                                                       |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `'MSE'`                  | [`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)                                          |
| `'L1'`                   | [`nn.L1Loss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)                                     |
| `'SmoothL1'`             | [`nn.SmoothL1Loss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)                         |
| `'SoftMarginLoss'`       | [`nn.SoftMarginLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html)                     |
| `'Huber'`                | [`nn.HuberLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)                               |
| `'CrossEntropy'`         | [`nn.CrossEntropyLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)                 |
| `'KLDiv'`                | [`nn.KLDivLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)                               |
| `'PoissonNLL'`           | [`nn.PoissonNLLLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html)                     |
| `'MultiLabelSoftMargin'` | [`nn.MultiLabelSoftMarginLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html) |

---

## Critères personnalisés

De nouveaux critères peuvent être enregistrés à l'exécution avec [`Container.add`](add.md) :

```python
from NeuralNetworks import crits
crits.add('MonCrit', mon_callable)   # signature : (net, inputs, outputs) -> Tensor
```

Cela permet notamment :

- des **PINNs** (Physics-Informed Neural Networks) en supervisé, via `torch.autograd.grad` sur `inputs` (`inputs.requires_grad_(True)`) ;
- de l'**apprentissage non supervisé**, en passant `outputs=None` au [`Trainer`](../Trainer/index.md) et en ignorant le troisième argument.