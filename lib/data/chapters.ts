export interface ChapterData {
    id: number;
    title: string;
    description: string;
    estimatedTime: string;
    objectives: string[];
    content: {
        type: "paragraph" | "heading" | "math-block" | "callout" | "code-note" | "interactive";
        text?: string;
        latex?: string;
        title?: string;
        variant?: "info" | "warning" | "tip";
        component?: string;
        props?: Record<string, any>;
    }[];
}

export const chaptersData: Record<string, ChapterData> = {
    // ... Chapter 1 and 2 (omitted for brevity in this update, focusing on Chap 3)
    "1": {
        id: 1,
        title: "Introduction",
        description: "Vue d'ensemble de l'apprentissage profond : définitions, importance et mécanismes d'apprentissage.",
        estimatedTime: "25 min",
        objectives: [
            "Comprendre l'apprentissage profond et sa relation avec le machine learning",
            "Définir un modèle comme une fonction paramétrée",
            "Comprendre le concept d'apprentissage supervisé",
        ],
        content: [
            { type: "paragraph", text: "L'apprentissage profond (Deep Learning) est un sous-ensemble de l'apprentissage automatique qui utilise des réseaux de neurones artificiels à plusieurs couches pour extraire progressivement des caractéristiques de plus haut niveau à partir de données brutes." },
            // ... (rest of chap 1 abbreviated for now, focusing on 3)
        ]
    },
    // We will populate Chapter 3 fully below
    "3": {
        id: 3,
        title: "Réseaux de Neurones Superficiels (Shallow Nets)",
        description: "Théorème d'approximation universelle : comment une seule couche cachée peut représenter n'importe quelle fonction continue.",
        estimatedTime: "60 min",
        objectives: [
            "Construire un réseau de neurones avec une couche cachée",
            "Comprendre le rôle des fonctions d'activation (ReLU)",
            "Visualiser comment les unités cachées se combinent pour former des fonctions complexes",
            "Comprendre le théorème d'approximation universelle"
        ],
        content: [
            { type: "heading", text: "3.1 Introduction" },
            { type: "paragraph", text: "Dans le chapitre précédent, nous avons étudié la régression linéaire, qui associe une entrée scalaire x à une sortie scalaire y par une simple équation linéaire : f(x) = φ₀ + φ₁x. Ce modèle est facile à adapter, mais il ne peut représenter que des droites. Or, la plupart des relations dans le monde réel (par exemple, la relation entre la taille d'un enfant et son âge) ne sont pas linéaires." },
            { type: "paragraph", text: "Pour modéliser ces données complexes, nous avons besoin d'une famille de fonctions plus flexible. C'est ici qu'interviennent les réseaux de neurones superficiels (shallow neural networks). Ces réseaux peuvent décrire des relations non linéaires complexes." },

            { type: "heading", text: "3.2 Réseaux de Neurones" },
            { type: "paragraph", text: "Un réseau de neurones superficiel avec une couche cachée prend une entrée scalaire x, la transforme via plusieurs 'unités cachées', et produit une sortie scalaire y. L'équation fondamentale est :" },
            { type: "math-block", latex: "y = f(x; \\phi) = \\phi_0 + \\sum_{j=1}^{D} v_j a(\\theta_{j0} + \\theta_{j1}x)" },
            { type: "paragraph", text: "Décomposons cette équation :" },
            { type: "paragraph", text: "- x est l'entrée." },
            { type: "paragraph", text: "- D est le nombre d'unités cachées (la 'largeur' du réseau)." },
            { type: "paragraph", text: "- θj0 (biais) et θj1 (pente) sont les paramètres de la j-ème unité cachée." },
            { type: "paragraph", text: "- a(·) est une fonction d'activation non linéaire." },
            { type: "paragraph", text: "- vj est le poids de sortie de la j-ème unité." },
            { type: "paragraph", text: "- φ0 est le biais final de sortie." },

            { type: "callout", variant: "info", title: "Note sur la terminologie", text: "Les variables h_j = a(θ_{j0} + θ_{j1}x) sont appelées 'activations cachées'. L'ensemble de ces activations forme la 'couche cachée'." },

            { type: "interactive", component: "NetworkDiagram" }, // SVG Diagram

            { type: "heading", text: "3.3 Fonctions d'Activation (ReLU)" },
            { type: "paragraph", text: "Pour que le réseau puisse modéliser des fonctions non linéaires, nous devons introduire une non-linéarité à travers la fonction d'activation a(·). Si a(·) était linéaire, l'ensemble du réseau s'effondrerait en une simple régression linéaire, peu importe le nombre d'unités !" },
            { type: "paragraph", text: "La fonction d'activation la plus courante en apprentissage profond moderne est l'unité linéaire rectifiée (Rectified Linear Unit), ou ReLU :" },
            { type: "math-block", latex: "\\text{ReLU}(z) = \\max(0, z)" },
            { type: "paragraph", text: "Graphiquement, cela ressemble à une 'charnière' :" },

            { type: "interactive", component: "ReLUHinge" }, // SVG Component

            { type: "heading", text: "3.4 Intuition : Construire avec des Charnières" },
            { type: "paragraph", text: "Chaque unité cachée j calcule une fonction h_j(x) forme de charnière. En combinant ces charnières, nous pouvons construire des fonctions linéaires par morceaux (piecewise linear)." },
            { type: "paragraph", text: "Considérons l'équation d'une seule unité :" },
            { type: "math-block", latex: "h(x) = v \\cdot \\text{ReLU}(\\theta_0 + \\theta_1 x)" },
            { type: "paragraph", text: "- θ1 contrôle la pente de la partie active." },
            { type: "paragraph", text: "- θ0 contrôle la position du point de cassure (le 'genou')." },
            { type: "paragraph", text: "- v contrôle l'échelle verticale et l'orientation (vers le haut ou le bas)." },

            { type: "paragraph", text: "Interagissez avec la visualisation ci-dessous pour voir comment la somme de seulement trois unités cachées (charnières colorées) peut créer une forme complexe (ligne bleue)." },

            { type: "interactive", component: "ShallowNetViz", props: { defaultUnits: 3 } },

            { type: "heading", text: "3.5 Théorème d'Approximation Universelle" },
            { type: "paragraph", text: "Le 'Théorème d'Approximation Universelle' stipule qu'un réseau de neurones avec une seule couche cachée pour décrire n'importe quelle fonction continue avec une précision arbitraire, à condition d'avoir suffisamment d'unités cachées." },
            { type: "paragraph", text: "C'est un résultat puissant : cela signifie que même l'architecture la plus simple est capable d'apprendre pratiquement n'importe quoi, si on lui donne assez de capacité (largeur D)." },

            { type: "heading", text: "3.6 Entrées et Sorties Multivariées" },
            { type: "paragraph", text: "Jusqu'à présent, nous avons considéré une entrée scalaire x et une sortie scalaire y. En réalité, les réseaux de neurones traitent souvent des vecteurs." },
            { type: "paragraph", text: "Pour une entrée vecteur x de dimension Di et une sortie vecteur y de dimension Do, avec une couche cachée de taille D :" },
            { type: "math-block", latex: "y_k = \\phi_{k0} + \\sum_{j=1}^{D} v_{kj} a\\left(\\theta_{j0} + \\sum_{i=1}^{D_i} \\theta_{ji} x_i\\right)" },
            { type: "paragraph", text: "Ou en notation matricielle, plus compacte et efficace pour les calculs :" },
            { type: "math-block", latex: "\\mathbf{y} = \\boldsymbol{\\phi} + \\mathbf{V} a(\\boldsymbol{\\theta}_0 + \\boldsymbol{\\Theta} \\mathbf{x})" },

            { type: "heading", text: "Exercices" },
            { type: "paragraph", text: "Testez votre compréhension avec ces exercices :" },
            { type: "callout", variant: "warning", title: "Exercice 3.1", text: "Démontrez que si la fonction d'activation a(z) est linéaire (a(z) = c*z), alors le réseau entier se réduit à une transformation linéaire." },
            { type: "callout", variant: "warning", title: "Exercice 3.2", text: "Combien de paramètres possède un réseau superficiel avec 10 entrées, 50 unités cachées et 3 sorties ?" },
        ]
    }
};
