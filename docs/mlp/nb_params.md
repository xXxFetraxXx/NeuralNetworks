# Nombre total de paramètres du MLP (`nb_params`)

Calcule le nombre total de paramètres (poids) du MLP.

Cette méthode parcourt les paramètres du modèle et ne compte que les **poids** des couches linéaires.  
Dans un `nn.Linear`, `parameters()` renvoie dans l'ordre :  
- les poids (indice pair)  
- puis les biais (indice impair)  

On ignore donc les biais dans ce comptage.

---

## Retour

| Type | Description |
| ---- | ----------- |
| `int` | Nombre total de paramètres pondérés (weights) dans le MLP. |