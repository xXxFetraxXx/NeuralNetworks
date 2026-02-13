# **Dynamique du learning rate**

---

Le learning rate a une double dépendance, il varie en fonction de l'époque mais aussi en fonction du résidu de l'époque.
Cela permet d'avoir un learning rate adapté en fonction de la taille de l'entrainement et qu'il continu à apprendre dans le cas ou les erreurs sont grandes.

La fusion des deux dynamiques de fait de la manière suivante:

$$
lr = (\text{init_lr} - \text{final_lr}) \cdot \max (\text{lr_epoch} , \text{lr_loss}) + \text{final_lr}
$$

???+ abstract "Dynamique en fonction de l'époque" 

    $$
    \text{lr_epoch} = \begin{cases}
            1 - \frac{1}{2} (6 \cdot (\frac{epoch}{0.1 \cdot \text{Nb_epochs}})^5 - 15 (\cdot \frac{epoch}{0.1 \cdot \text{Nb_epochs}})^4 + 10 \cdot (\frac{epoch}{0.1 \cdot \text{Nb_epochs}})^3) \quad &\text{si} \, epoch <= 0.1 \cdot \text{Nb_epochs}\\
            \frac{1}{2} (1 - 6 \cdot (\frac{epoch - 0.1 \cdot \text{Nb_epochs}}{0.9 \cdot \text{Nb_epochs}})^5 + 15 \cdot (\frac{epoch - 0.1 \cdot \text{Nb_epochs}}{0.9 \cdot \text{Nb_epochs}})^4 - 10 \cdot (\frac{epoch - 0.1 \cdot \text{Nb_epochs}}{0.9 \cdot \text{Nb_epochs}})^3) \quad &\text{si} \, epoch >= 0.1 \cdot \text{Nb_epochs}\\
        \end{cases}
    $$

    === "Échelle linéaire"
        ![epoch](epoch_b.png#only-light)
        ![epoch](epoch_w.png#only-dark)

    === "Échelle logarithmique"
        ![epoch_log](epoch_log_b.png#only-light)
        ![epoch_log](epoch_log_w.png#only-dark)

???+ abstract "Dynamique en fonction des résidus"

    $$
    \text{lr_loss} =\frac{1}{2} (1 + loss^4 + (loss^2 -2 \cdot loss +1)^4  )
    $$

    === "Échelle linéaire"
        ![res](res_b.png#only-light)
        ![res](res_w.png#only-dark)

    === "Échelle logarithmique"
        ![res_log](res_log_b.png#only-light)
        ![res_log](res_log_w.png#only-dark)