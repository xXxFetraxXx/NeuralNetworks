# Extraction des neurones du MLP (`neurons`)

Extrait l'ensemble des neurones (biais) du MLP couche par couche.

Cette méthode parcourt les paramètres du modèle et récupère uniquement les poids associés aux biais des couches linéaires.  
Dans un `nn.Linear`, `parameters()` renvoie successivement :  
- les poids (weights)  
- puis les biais (bias)  

Les indices impairs correspondent donc aux biais.

---

## Retour

| Type             | Description |
| ---------------- | ----------- |
| `list[np.ndarray]` | Liste contenant chaque neurone (chaque valeur de biais), converti en array NumPy sur CPU. |