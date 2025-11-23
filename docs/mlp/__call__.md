# Inférence avec le MLP (`__call__`)

Effectue une inférence complète : encodage éventuel, passage dans le MLP, puis retour en `numpy`.

Cette méthode permet d’utiliser l’objet comme une fonction directement.

---

## Paramètres

| Paramètre | Type       | Description                                                                                |
| --------- | ---------- | ------------------------------------------------------------------------------------------ |
| `x`       | array-like | Entrée(s) à prédire. Peut être un tableau numpy, une liste, ou déjà un tenseur compatible. |

---

## Retour

| Type         | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `np.ndarray` | Sortie du MLP après encodage et propagation avant, convertie en tableau numpy sur CPU. |