document.addEventListener("DOMContentLoaded", function() {
    // Fonction pour mettre à jour toutes les images switchables
    function updateFigures() {
        const isDark = document.body.classList.contains("md-dark");
        document.querySelectorAll(".theme-switchable").forEach(img => {
            const light = img.getAttribute("data-light");
            const dark = img.getAttribute("data-dark");
            if (light && dark) {
                img.src = isDark ? dark : light;
            }
        });
    }

    // Mettre à jour dès le chargement
    updateFigures();

    // Observer les changements de classe sur le body (quand toggle est utilisé)
    const observer = new MutationObserver(mutations => {
        mutations.forEach(() => updateFigures());
    });

    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
});