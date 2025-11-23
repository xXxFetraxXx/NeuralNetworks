# Construction du MLP (`Create_MLP`)

Construit un Multi-Layer Perceptron (MLP) standard composé :  
- d'une succession de couches Linéaires  
- suivies d'une normalisation (`self.norm`) après chaque couche cachée  
- et d'une activation Sigmoid sur la couche de sortie

---

## Paramètres

| Paramètre | Type       | Description |
| --------- | ---------- | ----------- |
| `layers`  | list[int]  | Liste des dimensions successives du réseau. Exemple : `[in_features, hidden1, hidden2, ..., out_features]` |

---

## Retour

| Type           | Description |
| -------------- | ----------- |
| `nn.Sequential` | Le MLP complet sous forme de séquence PyTorch |

---

## Notes

- La couche finale applique systématiquement une Sigmoid, adaptée à des sorties dans `[0, 1]`.