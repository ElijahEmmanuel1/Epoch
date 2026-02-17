export interface ChapterData {
    id: number;
    title: string;
    description: string;
    estimatedTime: string;
    objectives: string[];
    content: {
        type: "paragraph" | "heading" | "math-block" | "callout" | "code-note" | "interactive" | "list";
        text?: string;
        latex?: string;
        title?: string;
        variant?: "info" | "warning" | "tip";
        component?: string;
        props?: Record<string, any>;
        items?: string[]; // For lists
    }[];
}

export const chaptersData: Record<string, ChapterData> = {
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
            { type: "heading", text: "Qu'est-ce qu'un Modèle ?" },
            { type: "paragraph", text: "Nous pouvons décrire un modèle mathématiquement comme une fonction f(x; φ) qui associe une entrée x à une sortie y. Le point-virgule sépare les données d'entrée des paramètres du modèle." },
            { type: "callout", variant: "tip", title: "Concept Clé : Le Modèle", text: "L'objectif de l'apprentissage est de trouver les paramètres φ tels que le modèle produise des prédictions utiles :" },
            { type: "math-block", latex: "y \\approx f(x; \\phi)" },
            { type: "heading", text: "Apprentissage Supervisé" },
            { type: "paragraph", text: "En apprentissage supervisé, nous disposons d'un jeu de données d'entraînement composé de paires entrée-sortie {(xᵢ, yᵢ)}. L'objectif est d'apprendre une fonction qui se généralise bien aux données non vues — et pas simplement de mémoriser les exemples d'entraînement." },
            { type: "callout", variant: "info", title: "Fonction de Perte", text: "Pour mesurer la qualité de l'ajustement de notre modèle aux données, nous définissons une fonction de perte L(φ) qui quantifie l'écart entre les prédictions et la vérité terrain :" },
            { type: "math-block", latex: "L(\\phi) = \\sum_{i=1}^{N} \\ell\\big(f(x_i; \\phi),\\, y_i\\big)" },
            { type: "paragraph", text: "Le processus d'entraînement consiste à trouver les paramètres qui minimisent cette perte. Ce problème d'optimisation est au cœur de tout l'apprentissage automatique." },
        ]
    },
    "3": {
        id: 3,
        title: "Réseaux de Neurones Superficiels",
        description: "Les réseaux superficiels décrivent des fonctions linéaires par morceaux et sont assez expressifs pour approximer des relations arbitrairement complexes.",
        estimatedTime: "90 min",
        objectives: [
            "Comprendre l'architecture d'un réseau à une couche cachée",
            "Analyser le rôle de l'activation ReLU et des régions linéaires",
            "Saisir le théorème d'approximation universelle",
            "Étendre le modèle aux entrées et sorties multivariées"
        ],
        content: [
            { type: "paragraph", text: "Le chapitre 2 a introduit l'apprentissage supervisé à l'aide de la régression linéaire 1D. Cependant, ce modèle ne peut décrire la relation entrée/sortie que sous forme de ligne droite. Ce chapitre présente les réseaux de neurones superficiels. Ceux-ci décrivent des fonctions linéaires par morceaux et sont suffisamment expressifs pour approximer des relations arbitrairement complexes entre des entrées et des sorties multidimensionnelles." },

            { type: "heading", text: "3.1 Exemple de réseau de neurones" },
            { type: "paragraph", text: "Les réseaux de neurones superficiels sont des fonctions y = f[x, φ] avec des paramètres φ qui associent des entrées multivariées x à des sorties multivariées y. Nous reportons une définition complète à la section 3.4 et introduisons les idées principales à l'aide d'un exemple de réseau f[x, φ] qui associe une entrée scalaire x à une sortie scalaire y et possède dix paramètres φ = {φ0, φ1, φ2, φ3, θ10, θ11, θ20, θ21, θ30, θ31} :" },
            { type: "math-block", latex: "y = f[x, \\phi] = \\phi_0 + \\phi_1 a[\\theta_{10} + \\theta_{11} x] + \\phi_2 a[\\theta_{20} + \\theta_{21} x] + \\phi_3 a[\\theta_{30} + \\theta_{31} x]" },
            { type: "code-note", title: "Équation 3.1", text: "Cette équation est fondamentale. Elle représente la somme pondérée de trois unités cachées." },

            { type: "paragraph", text: "Nous pouvons décomposer ce calcul en trois parties : premièrement, nous calculons trois fonctions linéaires des données d'entrée (θ10 + θ11x, etc.). Deuxièmement, nous passons les trois résultats à travers une fonction d'activation a[•]. Enfin, nous pondérons les trois activations résultantes avec φ1, φ2 et φ3, nous les sommons et ajoutons un décalage (biais) φ0." },

            { type: "paragraph", text: "Pour compléter la description, nous devons définir la fonction d'activation a[•]. Il existe de nombreuses possibilités, mais le choix le plus courant est l'unité linéaire rectifiée ou ReLU (Rectified Linear Unit) :" },
            { type: "math-block", latex: "a[z] = \\text{ReLU}[z] = \\begin{cases} 0 & z < 0 \\\\ z & z \\ge 0 \\end{cases}" },

            { type: "paragraph", text: "Cela renvoie l'entrée lorsqu'elle est positive et zéro sinon." },

            { type: "interactive", component: "ReLUHinge" },

            { type: "heading", text: "3.1.1 Intuition des réseaux de neurones" },
            { type: "paragraph", text: "En fait, l'équation 3.1 représente une famille de fonctions linéaires par morceaux continues avec jusqu'à quatre régions linéaires. Pour rendre cela plus facile à comprendre, nous divisons la fonction en deux parties. Premièrement, nous introduisons les quantités intermédiaires :" },
            { type: "math-block", latex: "h_1 = a[\\theta_{10} + \\theta_{11} x], \\quad h_2 = a[\\theta_{20} + \\theta_{21} x], \\quad h_3 = a[\\theta_{30} + \\theta_{31} x]" },
            { type: "paragraph", text: "où nous désignons h1, h2 et h3 comme des unités cachées. Deuxièmement, nous calculons la sortie en combinant ces unités cachées avec une fonction linéaire :" },
            { type: "math-block", latex: "y = \\phi_0 + \\phi_1 h_1 + \\phi_2 h_2 + \\phi_3 h_3" },

            { type: "paragraph", text: "Chaque unité cachée contient une fonction linéaire de l'entrée qui est écrêtée par la fonction ReLU en dessous de zéro. Les positions où les trois lignes croisent zéro deviennent les trois 'joints' (points de cassure) dans la sortie finale." },

            { type: "interactive", component: "ShallowNetViz", props: { defaultUnits: 3 } },

            { type: "paragraph", text: "Ci-dessus, vous pouvez voir comment la somme pondérée de ces fonctions 'charnières' crée une fonction complexe linéaire par morceaux." },

            { type: "heading", text: "3.1.2 Représentation des réseaux de neurones" },
            { type: "paragraph", text: "Nous visualisons ce réseau comme suit : l'entrée est à gauche, les unités cachées sont au milieu et la sortie est à droite. Chaque connexion représente l'un des paramètres (pentes). Pour simplifier, nous ne dessinons généralement pas les paramètres d'interception (biais)." },

            { type: "interactive", component: "NetworkDiagram" },

            { type: "heading", text: "3.2 Théorème d'approximation universelle" },
            { type: "paragraph", text: "Le nombre d'unités cachées dans un réseau superficiel est une mesure de la capacité du réseau. Avec des fonctions d'activation ReLU, la sortie d'un réseau avec D unités cachées a au plus D joints et est donc une fonction linéaire par morceaux avec au plus D + 1 régions linéaires. À mesure que nous ajoutons plus d'unités cachées, le modèle peut approximer des fonctions plus complexes." },
            { type: "callout", variant: "tip", title: "Théorème", text: "Le théorème d'approximation universelle prouve que pour toute fonction continue, il existe un réseau superficiel (avec suffisamment d'unités cachées) qui peut approximer cette fonction avec n'importe quelle précision spécifiée." },

            { type: "heading", text: "3.3 Entrées et sorties multivariées" },
            { type: "paragraph", text: "Le théorème d'approximation universelle s'applique également au cas plus général où le réseau associe des entrées multivariées x de dimension Di à des prédictions de sortie multivariées y de dimension Do." },

            { type: "heading", text: "3.3.1 Visualisation des sorties multivariées" },
            { type: "paragraph", text: "Pour étendre le réseau aux sorties multivariées y, nous utilisons simplement une fonction linéaire différente des unités cachées pour chaque sortie. Ainsi, un réseau avec une entrée scalaire x, quatre unités cachées et une sortie multivariée 2D y = [y1, y2] serait défini comme :" },
            { type: "math-block", latex: "h_d = a[\\theta_{d0} + \\theta_{d1} x] \\quad \\text{pour } d \\in \\{1,..,4\\}" },
            { type: "paragraph", text: "et" },
            { type: "math-block", latex: "y_1 = \\phi_{10} + \\sum \\phi_{1d} h_d, \\quad y_2 = \\phi_{20} + \\sum \\phi_{2d} h_d" },

            { type: "heading", text: "3.3.2 Visualisation des entrées multivariées" },
            { type: "paragraph", text: "Pour gérer les entrées multivariées, nous étendons les relations linéaires entre l'entrée et les unités cachées. Chaque unité cachée reçoit maintenant une combinaison linéaire de toutes les entrées, ce qui forme un hyperplan orienté dans l'espace d'entrée." },
            { type: "math-block", latex: "h_d = a[\\theta_{d0} + \\sum_{i=1}^{D_i} \\theta_{di} x_i]" },

            { type: "heading", text: "3.4 Réseaux de neurones superficiels : cas général" },
            { type: "paragraph", text: "Nous définissons maintenant une équation générale pour un réseau de neurones superficiel y = f[x, φ] qui associe une entrée multidimensionnelle x ∈ R^Di à une sortie multidimensionnelle y ∈ R^Do en utilisant D unités cachées. Chaque unité cachée est calculée comme :" },
            { type: "math-block", latex: "h_d = a\\left[\\theta_{d0} + \\sum_{i=1}^{D_i} \\theta_{di} x_i\\right]" },
            { type: "paragraph", text: "et celles-ci sont combinées linéairement pour créer la sortie :" },
            { type: "math-block", latex: "y_j = \\phi_{j0} + \\sum_{d=1}^{D} \\phi_{jd} h_d" },

            { type: "heading", text: "3.5 Terminologie" },
            { type: "paragraph", text: "Les réseaux de neurones ont beaucoup de jargon associé. Ils sont souvent décrits en termes de couches :" },
            {
                type: "list", items: [
                    "Couche d'entrée (Input Layer) : Les variables x.",
                    "Couche cachée (Hidden Layer) : Les variables h (après activation).",
                    "Couche de sortie (Output Layer) : Les variables y."
                ]
            },
            { type: "paragraph", text: "Les unités cachées elles-mêmes sont parfois appelées 'neurones'. Les valeurs entrant dans les unités cachées (avant l'application de ReLU) sont appelées pré-activations. Les connexions représentent les poids (weights) et les décalages sont les biais (biases)." },

            { type: "heading", text: "3.6 Résumé" },
            { type: "paragraph", text: "Les réseaux de neurones superficiels ont une couche cachée. Ils (i) calculent plusieurs fonctions linéaires de l'entrée, (ii) passent chaque résultat par une fonction d'activation, puis (iii) prennent une combinaison linéaire de ces activations pour former les sorties." },

            { type: "heading", text: "Problèmes" },
            { type: "callout", variant: "warning", title: "Problème 3.1", text: "Quel type de mappage de l'entrée à la sortie serait créé si la fonction d'activation de l'équation 3.1 était linéaire de sorte que a[z] = ψ0 + ψ1z ? Quel type de mappage serait créé si la fonction d'activation était supprimée, donc a[z] = z ?" },
            { type: "callout", variant: "warning", title: "Problème 3.2", text: "Pour chacune des quatre régions linéaires de la figure, indiquez quelles unités cachées sont inactives et lesquelles sont actives (c'est-à-dire lesquelles écrêtent ou non leurs entrées)." },
            { type: "callout", variant: "warning", title: "Problème 3.3", text: "Dérivez des expressions pour les positions des 'joints' dans la fonction en termes des dix paramètres φ et de l'entrée x." },
            { type: "callout", variant: "warning", title: "Problème 3.5", text: "Prouvez que la propriété suivante est vraie pour α ∈ R+ : ReLU[α · z] = α · ReLU[z]. C'est ce qu'on appelle la propriété d'homogénéité non négative de la fonction ReLU." }
        ]
    }
};
