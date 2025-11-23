# Visualisation du MLP

Génère un aperçu visuel du MLP et affiche ses dimensions.

---

Cette méthode :

- Crée une version éventuellement "compressée" des dimensions du réseau  
  (utile lorsque certaines couches dépassent 30 neurones, afin de conserver une visualisation lisible)
- Utilise `visualtorch.graph_view` pour afficher un schéma du MLP
- Imprime la liste réelle des dimensions du réseau
- Retourne une chaîne indiquant si un redimensionnement a été appliqué

---

## Notes

- Le redimensionnement ne modifie pas le MLP réel. Il ne sert qu'à améliorer la lisibilité du graphe affiché.
- Si Fourier Features sont activées, seule la première dimension est recalculée en conséquence.
