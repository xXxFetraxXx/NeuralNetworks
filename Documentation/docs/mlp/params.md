# Extraction des poids du MLP (`params`)

Retourne la liste de tous les poids (weights) du MLP.

Cette fonction extrait uniquement les matrices de poids des couches linéaires, en ignorant les biais.  
Dans un `nn.Linear`, `parameters()` renvoie dans l'ordre :  
- les poids (indice pair)  
- les biais (indice impair)

La méthode :

- Parcourt les paramètres du réseau  
- Sélectionne uniquement ceux correspondant aux poids  
- Sépare chaque ligne de la matrice de poids  
- Convertit chaque ligne en `ndarray` CPU détaché

---

## Retour

| Type             | Description |
| ---------------- | ----------- |
| `list[np.ndarray]` | Liste contenant chaque ligne des matrices de poids (chaque élément est un vecteur numpy). |