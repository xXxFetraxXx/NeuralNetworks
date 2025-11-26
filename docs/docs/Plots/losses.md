# `losses`

Affiche les courbes de pertes (training loss) de plusieurs réseaux MLP.

---

## Paramètres

| Paramètre  | Type                | Description                                                                                                 |
| ---------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `*nets`    | MLP ou liste de MLP | Un ou plusieurs réseaux possédant un attribut `.losses` contenant l'historique des pertes (liste de float). |

---

## Notes

- L’axe X correspond aux itérations (epochs ou steps).  
- L’axe Y correspond à la valeur de la perte.  
- Utilise matplotlib en mode interactif pour un affichage dynamique.  
