export interface ChapterData {
    id: number;
    title: string;
    description: string;
    estimatedTime: string;
    objectives: string[];
    content: {
        type: "paragraph" | "heading" | "math-block" | "callout" | "code-note" | "interactive" | "list" | "code";
        text?: string;
        latex?: string;
        title?: string;
        variant?: "info" | "warning" | "tip";
        component?: string;
        props?: Record<string, any>;
        items?: string[]; // For lists
        // Code block fields
        language?: string;
        initialCode?: string;
        expectedOutput?: string;
        description?: string;
        solution?: string;
        hints?: string[];
    }[];
}

export const chaptersData: Record<string, ChapterData> = {
    "1": {
        id: 1,
        title: "Introduction",
        description: "Vue d'ensemble de l'apprentissage profond : définitions, importance, mécanismes d'apprentissage, et premiers pas en code.",
        estimatedTime: "45 min",
        objectives: [
            "Comprendre l'apprentissage profond et sa relation avec le machine learning et l'IA",
            "Définir un modèle comme une fonction paramétrée f(x; φ)",
            "Comprendre le concept d'apprentissage supervisé, non supervisé et par renforcement",
            "Manipuler des tenseurs et opérations basiques en Python/NumPy",
            "Implémenter un modèle linéaire simple de bout en bout",
        ],
        content: [
            // ─── SECTION 1: Qu'est-ce que le Deep Learning ? ───
            { type: "heading", text: "1.1 Qu'est-ce que le Deep Learning ?" },
            { type: "paragraph", text: "L'intelligence artificielle (IA) est le domaine qui vise à créer des machines capables de réaliser des tâches requérant habituellement l'intelligence humaine. Le machine learning (ML) est un sous-ensemble de l'IA où les systèmes apprennent à partir de données plutôt que d'être explicitement programmés. Le deep learning est un sous-ensemble du ML qui utilise des réseaux de neurones à couches multiples." },
            { type: "callout", variant: "info", title: "Hiérarchie des concepts", text: "IA ⊃ Machine Learning ⊃ Deep Learning ⊃ Réseaux de Neurones Profonds. Le deep learning se distingue par la profondeur de ses architectures : plusieurs couches de transformations non linéaires successives." },
            { type: "paragraph", text: "Depuis 2012 (AlexNet), le deep learning domine en vision par ordinateur, traitement du langage naturel, reconnaissance vocale, et bien d'autres domaines. Ces succès reposent sur trois piliers : (1) de grandes quantités de données, (2) une puissance de calcul accrue (GPU/TPU), et (3) des innovations algorithmiques (ReLU, Dropout, BatchNorm, Transformers)." },

            { type: "code", title: "Vérifier votre environnement", language: "python", description: "Commençons par vérifier que Python et les bibliothèques essentielles sont disponibles. Ce code simule les imports de base.", initialCode: "# Vérification de l'environnement\npython_version = 3.11\nnumpy_available = 1\ntorch_available = 1\n\nprint(f\"Python {python_version}\")\nprint(f\"NumPy: {'OK' if numpy_available else 'MANQUANT'}\")\nprint(f\"PyTorch: {'OK' if torch_available else 'MANQUANT'}\")", expectedOutput: "Python 3.11\nNumPy: OK\nPyTorch: OK", hints: ["Assurez-vous d'avoir Python 3.8+ installé", "pip install numpy torch"] },

            // ─── SECTION 2: Types d'apprentissage ───
            { type: "heading", text: "1.2 Les paradigmes d'apprentissage" },
            { type: "paragraph", text: "Il existe trois paradigmes principaux en machine learning, chacun correspondant à un type différent de signal d'apprentissage :" },
            { type: "list", items: [
                "Apprentissage supervisé : On dispose de paires (entrée, sortie attendue). Le modèle apprend une fonction qui associe les entrées aux sorties. Exemples : classification d'images, prédiction de prix.",
                "Apprentissage non supervisé : Pas de labels. Le modèle cherche des structures cachées dans les données. Exemples : clustering, réduction de dimensionnalité, modèles génératifs.",
                "Apprentissage par renforcement : Un agent interagit avec un environnement et apprend à maximiser une récompense cumulative. Exemples : jeux, robotique, navigation.",
            ] },
            { type: "callout", variant: "tip", title: "Focus : Apprentissage supervisé", text: "Ce livre se concentre principalement sur l'apprentissage supervisé, qui est le paradigme le plus mature et le plus utilisé en pratique. Nous y revenons en détail ci-dessous." },

            // ─── SECTION 3: Modèle mathématique ───
            { type: "heading", text: "1.3 Le modèle comme fonction paramétrée" },
            { type: "paragraph", text: "En apprentissage supervisé, nous cherchons une fonction f qui, étant donné une entrée x, produit une prédiction ŷ la plus proche possible de la vraie sortie y. Cette fonction possède des paramètres φ qu'il faut apprendre :" },
            { type: "math-block", latex: "\\hat{y} = f(x; \\phi)" },
            { type: "paragraph", text: "Les paramètres φ encodent tout ce que le modèle a appris. Pour un réseau de neurones, ce sont les poids et les biais. Le nombre de paramètres peut aller de quelques dizaines (régression linéaire) à plusieurs milliards (GPT, LLaMA)." },
            { type: "callout", variant: "info", title: "Notation", text: "On utilise le point-virgule dans f(x; φ) pour séparer les données d'entrée x (variables) des paramètres φ (constants une fois entraînés). Certains auteurs écrivent f_φ(x) ou f(x, φ)." },

            { type: "code", title: "Modèle linéaire simple", language: "python", description: "Implémentez un modèle linéaire y = wx + b. Calculez la prédiction pour x=3 avec w=2.5 et b=1.0.", initialCode: "# Modèle linéaire : y = w*x + b\nw = 2.5\nb = 1.0\nx = 3\n\ny = w * x + b\nprint(y)", expectedOutput: "8.5", solution: "# Le modèle linéaire est la forme la plus simple\n# y = wx + b\n# avec w = pente (weight) et b = ordonnée à l'origine (bias)\nw = 2.5\nb = 1.0\nx = 3\n\ny = w * x + b\nprint(y)  # 2.5 * 3 + 1.0 = 8.5", hints: ["Un modèle linéaire a la forme y = wx + b", "Multipliez w par x, puis ajoutez b"] },

            // ─── SECTION 4: Fonction de perte ───
            { type: "heading", text: "1.4 Fonction de perte et entraînement" },
            { type: "paragraph", text: "Pour mesurer la qualité de notre modèle, on définit une fonction de perte (loss function) L(φ) qui quantifie l'écart entre les prédictions du modèle et les vraies valeurs. Plus la perte est faible, meilleur est le modèle." },
            { type: "math-block", latex: "L(\\phi) = \\frac{1}{N} \\sum_{i=1}^{N} \\ell\\big(f(x_i; \\phi),\\, y_i\\big)" },
            { type: "paragraph", text: "La perte la plus courante pour la régression est l'erreur quadratique moyenne (Mean Squared Error, MSE) :" },
            { type: "math-block", latex: "\\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (\\hat{y}_i - y_i)^2" },
            { type: "paragraph", text: "L'entraînement consiste à trouver les paramètres φ* qui minimisent cette perte. C'est un problème d'optimisation qu'on résout généralement par descente de gradient (voir Chapitre 6)." },
            { type: "math-block", latex: "\\phi^* = \\arg\\min_{\\phi} L(\\phi)" },

            { type: "code", title: "Calculer la MSE", language: "python", description: "Étant donné des prédictions et des valeurs réelles, calculez l'erreur quadratique moyenne (MSE).", initialCode: "# Données\npredictions = [2.5, 0.0, 2.1, 7.8]\ntargets     = [3.0, -0.5, 2.0, 7.5]\n\n# Calculez la MSE\nn = 4\nerror1 = (2.5 - 3.0) ** 2\nerror2 = (0.0 - -0.5) ** 2\nerror3 = (2.1 - 2.0) ** 2\nerror4 = (7.8 - 7.5) ** 2\n\nmse = (error1 + error2 + error3 + error4) / n\nprint(round(mse, 4))", expectedOutput: "0.1275", solution: "# MSE = (1/N) * Σ(ŷᵢ - yᵢ)²\npredictions = [2.5, 0.0, 2.1, 7.8]\ntargets     = [3.0, -0.5, 2.0, 7.5]\n\nn = 4\nerror1 = (2.5 - 3.0) ** 2  # 0.25\nerror2 = (0.0 - -0.5) ** 2  # 0.25\nerror3 = (2.1 - 2.0) ** 2   # 0.01\nerror4 = (7.8 - 7.5) ** 2   # 0.09\n\nmse = (error1 + error2 + error3 + error4) / n\nprint(round(mse, 4))  # 0.1275", hints: ["MSE = moyenne des carrés des erreurs", "Erreur = prédiction - valeur réelle", "N'oubliez pas de diviser par le nombre d'exemples N"] },

            // ─── SECTION 5: Tenseurs ───
            { type: "heading", text: "1.5 Tenseurs : la brique de base" },
            { type: "paragraph", text: "Les réseaux de neurones opèrent sur des tenseurs — des tableaux multidimensionnels de nombres. Un scalaire est un tenseur de rang 0, un vecteur de rang 1, une matrice de rang 2, et ainsi de suite. Comprendre les tenseurs est essentiel car toutes les données (images, texte, audio) sont représentées comme des tenseurs." },
            { type: "list", items: [
                "Scalaire (rang 0) : un seul nombre, ex: la température t = 23.5",
                "Vecteur (rang 1) : une liste de nombres, ex: un pixel RGB = [255, 128, 0]",
                "Matrice (rang 2) : un tableau 2D, ex: une image en niveaux de gris 28×28",
                "Tenseur 3D (rang 3) : ex: une image couleur 224×224×3",
                "Tenseur 4D (rang 4) : ex: un batch de 32 images = 32×224×224×3",
            ] },

            { type: "code", title: "Opérations sur les tenseurs", language: "python", description: "Simulez des opérations basiques sur des tenseurs : addition, multiplication, produit scalaire.", initialCode: "# Vecteurs (simulés comme listes)\na = [1, 2, 3]\nb = [4, 5, 6]\n\n# Produit scalaire (dot product)\n# dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]\ndot = 1*4 + 2*5 + 3*6\nprint(dot)\n\n# Norme au carré de a\nnorm_sq = 1**2 + 2**2 + 3**2\nprint(norm_sq)", expectedOutput: "32\n14", solution: "# Produit scalaire : Σ aᵢbᵢ\na = [1, 2, 3]\nb = [4, 5, 6]\n\ndot = 1*4 + 2*5 + 3*6  # = 4 + 10 + 18 = 32\nprint(dot)\n\n# Norme² = Σ aᵢ²\nnorm_sq = 1**2 + 2**2 + 3**2  # = 1 + 4 + 9 = 14\nprint(norm_sq)", hints: ["Le produit scalaire multiplie élément par élément puis somme", "La norme au carré est le produit scalaire d'un vecteur avec lui-même"] },

            // ─── SECTION 6: Workflow ───
            { type: "heading", text: "1.6 Le workflow du Machine Learning" },
            { type: "paragraph", text: "Un projet de ML suit un cycle structuré qui va de la collecte des données au déploiement du modèle :" },
            { type: "list", items: [
                "1. Collecte et préparation des données (nettoyage, normalisation, augmentation)",
                "2. Séparation en ensembles d'entraînement, de validation et de test",
                "3. Choix de l'architecture du modèle (hyperparamètres)",
                "4. Entraînement : optimisation des paramètres sur les données d'entraînement",
                "5. Évaluation : mesure des performances sur les données de validation/test",
                "6. Ajustement des hyperparamètres et retour à l'étape 3 si nécessaire",
                "7. Déploiement et monitoring en production",
            ] },
            { type: "callout", variant: "tip", title: "Train / Validation / Test", text: "Il est crucial de ne JAMAIS utiliser les données de test pendant l'entraînement ou la sélection du modèle. Le test set est réservé à l'évaluation finale. Utiliser le validation set pour ajuster les hyperparamètres." },

            { type: "code", title: "Séparation Train/Test", language: "python", description: "Séparez un dataset de 100 exemples en 80% train et 20% test. Calculez les tailles.", initialCode: "# Dataset de 100 exemples\nn_total = 100\ntrain_ratio = 0.8\n\nn_train = round(n_total * train_ratio)\nn_test = n_total - n_train\n\nprint(n_train)\nprint(n_test)", expectedOutput: "80\n20", hints: ["Multipliez le total par le ratio pour obtenir la taille du train set"] },

            // ─── SECTION 7: Résumé ───
            { type: "heading", text: "1.7 Résumé" },
            { type: "paragraph", text: "Le deep learning est un sous-domaine du machine learning qui utilise des réseaux de neurones à couches multiples. Un modèle est une fonction paramétrée f(x; φ) dont les paramètres sont appris en minimisant une fonction de perte sur des données d'entraînement. Toutes les données sont représentées comme des tenseurs. Le chapitre suivant introduit le modèle le plus simple : la régression linéaire." },

            { type: "heading", text: "Exercices" },
            { type: "callout", variant: "warning", title: "Exercice 1.1", text: "Quel est le nombre de paramètres d'un modèle linéaire y = Wx + b qui prend un vecteur x de dimension 10 et produit un vecteur y de dimension 3 ? (Comptez W et b séparément)" },
            { type: "code", title: "Exercice 1.1 — Compter les paramètres", language: "python", description: "Calculez le nombre total de paramètres pour un modèle linéaire avec entrée dim=10, sortie dim=3.", initialCode: "# Modèle linéaire : y = W @ x + b\n# W est une matrice de forme (sortie, entrée)\n# b est un vecteur de forme (sortie,)\n\ninput_dim = 10\noutput_dim = 3\n\nw_params = input_dim * output_dim\nb_params = output_dim\ntotal = w_params + b_params\n\nprint(total)", expectedOutput: "33", solution: "# W: matrice (3, 10) → 30 paramètres\n# b: vecteur (3,) → 3 paramètres\n# Total: 33 paramètres\n\ninput_dim = 10\noutput_dim = 3\n\nw_params = input_dim * output_dim  # 30\nb_params = output_dim               # 3\ntotal = w_params + b_params          # 33\n\nprint(total)", hints: ["W a une ligne par sortie et une colonne par entrée", "b a un élément par sortie"] },

            { type: "callout", variant: "warning", title: "Exercice 1.2", text: "Si un modèle a une MSE de 0.5 sur le train set et 2.0 sur le test set, que pouvez-vous en conclure ? Ce phénomène est appelé..." },
        ]
    },
    "3": {
        id: 3,
        title: "Réseaux de Neurones Superficiels",
        description: "Les réseaux superficiels décrivent des fonctions linéaires par morceaux. Avec une seule couche cachée, ils peuvent approximer n'importe quelle fonction continue.",
        estimatedTime: "120 min",
        objectives: [
            "Comprendre l'architecture d'un réseau à une couche cachée",
            "Analyser le rôle de l'activation ReLU et des régions linéaires",
            "Saisir le théorème d'approximation universelle",
            "Implémenter un réseau superficiel en Python",
            "Étendre le modèle aux entrées et sorties multivariées",
        ],
        content: [
            { type: "paragraph", text: "Le chapitre 2 a introduit l'apprentissage supervisé à l'aide de la régression linéaire 1D. Cependant, ce modèle ne peut décrire la relation entrée/sortie que sous forme de ligne droite. Ce chapitre présente les réseaux de neurones superficiels. Ceux-ci décrivent des fonctions linéaires par morceaux et sont suffisamment expressifs pour approximer des relations arbitrairement complexes entre des entrées et des sorties multidimensionnelles." },

            // ─── 3.1 Exemple de réseau ───
            { type: "heading", text: "3.1 Exemple de réseau de neurones" },
            { type: "paragraph", text: "Les réseaux de neurones superficiels sont des fonctions y = f[x, φ] avec des paramètres φ qui associent des entrées multivariées x à des sorties multivariées y. Nous introduisons les idées principales à l'aide d'un exemple de réseau f[x, φ] qui associe une entrée scalaire x à une sortie scalaire y et possède dix paramètres φ = {φ₀, φ₁, φ₂, φ₃, θ₁₀, θ₁₁, θ₂₀, θ₂₁, θ₃₀, θ₃₁} :" },
            { type: "math-block", latex: "y = f[x, \\phi] = \\phi_0 + \\phi_1 a[\\theta_{10} + \\theta_{11} x] + \\phi_2 a[\\theta_{20} + \\theta_{21} x] + \\phi_3 a[\\theta_{30} + \\theta_{31} x]" },

            { type: "paragraph", text: "Nous décomposons ce calcul en trois parties : premièrement, nous calculons trois fonctions linéaires des données d'entrée (θ₁₀ + θ₁₁x, etc.). Deuxièmement, nous passons les résultats à travers une fonction d'activation a[•]. Enfin, nous pondérons les trois activations avec φ₁, φ₂, φ₃, nous les sommons et ajoutons un biais φ₀." },

            { type: "paragraph", text: "La fonction d'activation la plus courante est l'unité linéaire rectifiée (ReLU) :" },
            { type: "math-block", latex: "a[z] = \\text{ReLU}[z] = \\max(0,\\, z)" },
            { type: "paragraph", text: "Elle renvoie l'entrée lorsqu'elle est positive et zéro sinon." },

            { type: "interactive", component: "ReLUHinge" },

            { type: "code", title: "Implémenter ReLU", language: "python", description: "Implémentez la fonction ReLU et testez-la sur plusieurs valeurs.", initialCode: "# Implémentez ReLU\n# ReLU(z) = max(0, z)\n\nz1 = -3.0\nz2 = 0.0\nz3 = 2.5\n\nrelu_z1 = max(0, z1)\nrelu_z2 = max(0, z2)\nrelu_z3 = max(0, z3)\n\nprint(relu_z1)\nprint(relu_z2)\nprint(relu_z3)", expectedOutput: "0\n0\n2.5", solution: "# ReLU(z) = max(0, z)\n# Valeurs négatives → 0\n# Valeurs positives → inchangées\n\nz1 = -3.0\nz2 = 0.0\nz3 = 2.5\n\nrelu_z1 = max(0, z1)  # 0\nrelu_z2 = max(0, z2)  # 0\nrelu_z3 = max(0, z3)  # 2.5\n\nprint(relu_z1)\nprint(relu_z2)\nprint(relu_z3)", hints: ["ReLU éteint les valeurs négatives", "Utilisez la fonction max() de Python"] },

            // ─── 3.1.1 Intuition ───
            { type: "heading", text: "3.1.1 Intuition des réseaux de neurones" },
            { type: "paragraph", text: "L'équation 3.1 représente une famille de fonctions linéaires par morceaux continues avec jusqu'à quatre régions linéaires. Les quantités intermédiaires sont les unités cachées :" },
            { type: "math-block", latex: "h_1 = a[\\theta_{10} + \\theta_{11} x], \\quad h_2 = a[\\theta_{20} + \\theta_{21} x], \\quad h_3 = a[\\theta_{30} + \\theta_{31} x]" },
            { type: "paragraph", text: "La sortie est ensuite :" },
            { type: "math-block", latex: "y = \\phi_0 + \\phi_1 h_1 + \\phi_2 h_2 + \\phi_3 h_3" },

            { type: "paragraph", text: "Chaque unité cachée crée une 'charnière' — un point où la pente change. La position de cette charnière dépend de θ_{d0} et θ_{d1}. En combinant plusieurs charnières avec des poids différents, le réseau peut créer n'importe quelle forme linéaire par morceaux." },

            { type: "callout", variant: "tip", title: "Intuition géométrique (Géron)", text: "Pensez à chaque neurone ReLU comme un interrupteur conditionnel : il « s'allume » quand son entrée dépasse un seuil et transmet le signal proportionnellement. La couche de sortie combine ces signaux, un peu comme un mélangeur audio qui ajuste le volume de chaque canal." },

            { type: "interactive", component: "ShallowNetViz", props: { defaultUnits: 3 } },

            { type: "code", title: "Forward pass d'un réseau superficiel", language: "python", description: "Implémentez le forward pass complet d'un réseau à 3 unités cachées ReLU pour x = 1.5", initialCode: "# Paramètres du réseau\ntheta_10 = -1.0; theta_11 = 2.0\ntheta_20 = 0.5;  theta_21 = -1.0\ntheta_30 = -0.5; theta_31 = 0.5\n\nphi_0 = 0.2; phi_1 = 1.0; phi_2 = -0.5; phi_3 = 0.8\n\nx = 1.5\n\n# Pré-activations\nz1 = theta_10 + theta_11 * x\nz2 = theta_20 + theta_21 * x\nz3 = theta_30 + theta_31 * x\n\n# Activations (ReLU)\nh1 = max(0, z1)\nh2 = max(0, z2)\nh3 = max(0, z3)\n\n# Sortie\ny = phi_0 + phi_1 * h1 + phi_2 * h2 + phi_3 * h3\n\nprint(h1)\nprint(h2)\nprint(h3)\nprint(y)", expectedOutput: "2.0\n0\n0.25\nprint: 2.4", solution: "# Forward pass étape par étape\n# 1) Pré-activations: z = θ₀ + θ₁·x\nz1 = -1.0 + 2.0 * 1.5   # = 2.0\nz2 = 0.5 + (-1.0) * 1.5  # = -1.0\nz3 = -0.5 + 0.5 * 1.5    # = 0.25\n\n# 2) ReLU: h = max(0, z)\nh1 = max(0, 2.0)   # = 2.0\nh2 = max(0, -1.0)  # = 0 (éteint!)\nh3 = max(0, 0.25)  # = 0.25\n\n# 3) Sortie: y = φ₀ + Σφᵢhᵢ\ny = 0.2 + 1.0*2.0 + (-0.5)*0 + 0.8*0.25\n# = 0.2 + 2.0 + 0 + 0.2 = 2.4\nprint(y)", hints: ["Calculez d'abord les pré-activations z = θ₀ + θ₁·x", "Appliquez ReLU: si z < 0, le résultat est 0", "La sortie est la combinaison linéaire des activations"] },

            // ─── 3.1.2 Représentation ───
            { type: "heading", text: "3.1.2 Représentation des réseaux de neurones" },
            { type: "paragraph", text: "Nous visualisons un réseau comme un graphe : l'entrée est à gauche, les unités cachées au milieu, et la sortie à droite. Chaque connexion (arête) représente un paramètre (poids). Les biais sont implicites." },
            { type: "interactive", component: "NetworkDiagram" },

            // ─── 3.2 Approximation universelle ───
            { type: "heading", text: "3.2 Théorème d'approximation universelle" },
            { type: "paragraph", text: "Le nombre d'unités cachées D détermine la capacité du réseau. Avec des ReLU, un réseau à D unités cachées crée au plus D+1 régions linéaires. Plus on ajoute d'unités, plus la fonction approximée peut être complexe." },
            { type: "callout", variant: "tip", title: "Théorème d'approximation universelle", text: "Pour toute fonction continue f définie sur un compact, et pour toute tolérance ε > 0, il existe un réseau de neurones superficiel (avec suffisamment d'unités cachées) tel que la distance entre le réseau et f est inférieure à ε. Ce résultat est existentiel : il garantit l'existence mais ne dit rien sur le nombre d'unités nécessaires ni sur la facilité de trouver les bons paramètres." },

            { type: "paragraph", text: "En pratique, le théorème signifie que les réseaux superficiels ont le pouvoir expressif nécessaire, mais ils peuvent nécessiter un nombre astronomique de neurones pour certaines fonctions. C'est l'une des motivations des réseaux profonds (Chapitre 4)." },

            { type: "code", title: "Approximation par morceaux", language: "python", description: "Un réseau superficiel avec D unités cachées ReLU crée au plus D+1 régions linéaires. Calculez le nombre de régions pour différentes largeurs.", initialCode: "# Nombre de régions linéaires max = D + 1\n# où D est le nombre d'unités cachées\n\nd_values = [3, 10, 50, 100, 1000]\n\nregions_3 = 3 + 1\nregions_10 = 10 + 1\nregions_50 = 50 + 1\nregions_100 = 100 + 1\nregions_1000 = 1000 + 1\n\nprint(regions_3)\nprint(regions_10)\nprint(regions_50)\nprint(regions_100)\nprint(regions_1000)", expectedOutput: "4\n11\n51\n101\n1001", hints: ["Le nombre maximum de régions est toujours D + 1"] },

            // ─── 3.3 Multivariée ───
            { type: "heading", text: "3.3 Entrées et sorties multivariées" },
            { type: "paragraph", text: "Le cas général du réseau superficiel associe un vecteur d'entrée x ∈ ℝ^Dᵢ à un vecteur de sortie y ∈ ℝ^Dₒ via D unités cachées. Chaque unité cachée reçoit une combinaison linéaire de toutes les entrées :" },
            { type: "math-block", latex: "h_d = a\\left[\\theta_{d0} + \\sum_{i=1}^{D_i} \\theta_{di} x_i\\right]" },
            { type: "paragraph", text: "Et les sorties combinent linéairement les activations :" },
            { type: "math-block", latex: "y_j = \\phi_{j0} + \\sum_{d=1}^{D} \\phi_{jd} h_d" },

            { type: "callout", variant: "info", title: "Perspective scikit-learn (Géron)", text: "En scikit-learn, un MLPRegressor ou MLPClassifier avec une seule couche cachée correspond exactement à ce réseau superficiel. Par défaut, scikit-learn utilise l'activation ReLU et la descente de gradient stochastique (SGD) pour l'optimiser." },

            { type: "code", title: "Compter les paramètres", language: "python", description: "Calculez le nombre total de paramètres d'un réseau superficiel avec Dᵢ=5 entrées, D=20 unités cachées, et Dₒ=3 sorties.", initialCode: "# Réseau superficiel\nDi = 5   # dimension entrée\nD = 20   # unités cachées\nDo = 3   # dimension sortie\n\n# Couche cachée : D * (Di + 1) params (poids + biais)\nhidden_params = D * (Di + 1)\n\n# Couche sortie : Do * (D + 1) params\noutput_params = Do * (D + 1)\n\ntotal = hidden_params + output_params\nprint(hidden_params)\nprint(output_params)\nprint(total)", expectedOutput: "120\n63\n183", solution: "# Couche cachée: chaque unité a Di poids + 1 biais\n# → D × (Di + 1) = 20 × 6 = 120\n\n# Couche sortie: chaque sortie a D poids + 1 biais\n# → Do × (D + 1) = 3 × 21 = 63\n\n# Total: 120 + 63 = 183 paramètres\n\nDi = 5; D = 20; Do = 3\nhidden_params = D * (Di + 1)    # 120\noutput_params = Do * (D + 1)    # 63\ntotal = hidden_params + output_params  # 183\nprint(hidden_params)\nprint(output_params)\nprint(total)", hints: ["Chaque unité cachée a Di poids d'entrée + 1 biais", "Chaque unité de sortie a D poids + 1 biais", "N'oubliez pas les biais !"] },

            // ─── 3.4 Forme matricielle ───
            { type: "heading", text: "3.4 Réseaux superficiels : forme matricielle" },
            { type: "paragraph", text: "Pour plus de concision, on écrit le réseau en notation matricielle :" },
            { type: "math-block", latex: "\\mathbf{h} = a\\left[\\boldsymbol{\\beta}_0 + \\boldsymbol{\\Omega}_0 \\mathbf{x}\\right]" },
            { type: "math-block", latex: "\\mathbf{y} = \\boldsymbol{\\beta}_1 + \\boldsymbol{\\Omega}_1 \\mathbf{h}" },
            { type: "paragraph", text: "où β₀ ∈ ℝ^D est le vecteur de biais de la couche cachée, Ω₀ ∈ ℝ^(D×Dᵢ) est la matrice de poids de la couche cachée, β₁ ∈ ℝ^Dₒ est le vecteur de biais de sortie, et Ω₁ ∈ ℝ^(Dₒ×D) est la matrice de poids de sortie." },

            { type: "code", title: "Forward pass en notation matricielle", language: "python", description: "Simulez un forward pass matriciel. Entrée x=[1, 2], 2 unités cachées, 1 sortie.", initialCode: "# Paramètres (simplifiés)\n# Couche cachée: 2 entrées → 2 unités cachées\n# W0 = [[0.5, -0.3], [0.2, 0.8]]\n# b0 = [-0.1, 0.3]\n\n# Entrée\nx1 = 1; x2 = 2\n\n# Pré-activations\nz1 = 0.5 * x1 + -0.3 * x2 + -0.1\nz2 = 0.2 * x1 + 0.8 * x2 + 0.3\n\n# ReLU\nh1 = max(0, z1)\nh2 = max(0, z2)\n\n# Sortie: W1 = [0.7, -0.4], b1 = 0.1\ny = 0.7 * h1 + -0.4 * h2 + 0.1\n\nprint(round(z1, 1))\nprint(round(z2, 1))\nprint(h1)\nprint(round(h2, 1))\nprint(round(y, 2))", expectedOutput: "-0.2\n2.1\n0\n2.1\n-0.74", hints: ["z = W·x + b pour chaque couche", "N'oubliez pas d'appliquer ReLU entre les couches"] },

            // ─── 3.5 Autres activations ───
            { type: "heading", text: "3.5 Fonctions d'activation alternatives" },
            { type: "paragraph", text: "Bien que ReLU soit la plus populaire, il existe d'autres fonctions d'activation, chacune avec ses avantages :" },
            { type: "list", items: [
                "Sigmoïde σ(z) = 1/(1+e^{-z}) : écrase la sortie entre 0 et 1. Utile en sortie pour la classification binaire, mais souffre du vanishing gradient.",
                "Tanh(z) = (e^z - e^{-z})/(e^z + e^{-z}) : sortie entre -1 et 1. Centrée autour de zéro, convergence souvent plus rapide que sigmoïde.",
                "Leaky ReLU : max(αz, z) avec α petit (ex: 0.01). Évite le problème des 'neurones morts' de ReLU.",
                "GELU (Gaussian Error Linear Unit) : utilisée dans les Transformers (BERT, GPT). Approximation douce de ReLU.",
                "Swish/SiLU : z·σ(z). Auto-porte. Fonctionne bien dans les réseaux profonds.",
            ] },

            { type: "code", title: "Comparer les activations", language: "python", description: "Calculez la sortie de différentes fonctions d'activation pour z = -1.0 et z = 2.0.", initialCode: "# ReLU\nrelu_neg = max(0, -1.0)\nrelu_pos = max(0, 2.0)\nprint(relu_neg)\nprint(relu_pos)\n\n# Leaky ReLU (alpha = 0.01)\nalpha = 0.01\nleaky_neg = max(alpha * -1.0, -1.0)\nleaky_pos = max(alpha * 2.0, 2.0)\n\n# Pour z < 0: leaky = alpha * z\nleaky_neg = alpha * -1.0\nprint(leaky_neg)\nprint(leaky_pos)", expectedOutput: "0\n2.0\n-0.01\n2.0", hints: ["ReLU: max(0, z)", "Leaky ReLU: si z >= 0 → z, sinon → α·z"] },

            // ─── 3.6 Terminologie ───
            { type: "heading", text: "3.6 Terminologie" },
            { type: "paragraph", text: "Les réseaux de neurones ont beaucoup de jargon associé :" },
            { type: "list", items: [
                "Couche d'entrée (Input Layer) : Les variables x.",
                "Couche cachée (Hidden Layer) : Les variables h (après activation).",
                "Couche de sortie (Output Layer) : Les variables y.",
                "Neurone : Synonyme d'unité cachée.",
                "Pré-activation : Valeur avant application de la fonction d'activation.",
                "Poids (Weights) : Les paramètres des connexions entre couches.",
                "Biais (Bias) : Le terme constant ajouté avant l'activation.",
                "Largeur (Width) : Nombre d'unités dans une couche cachée.",
                "Capacité : Mesure de la complexité des fonctions que le réseau peut représenter.",
            ] },

            // ─── 3.7 Implémentation PyTorch ───
            { type: "heading", text: "3.7 Implémentation en PyTorch" },
            { type: "paragraph", text: "En PyTorch (Godoy, Ch. 1-2), un réseau superficiel se définit en quelques lignes. La bibliothèque gère automatiquement les tenseurs, le calcul des gradients, et l'optimisation." },
            { type: "callout", variant: "info", title: "PyTorch vs NumPy", text: "PyTorch ressemble beaucoup à NumPy, mais ajoute : (1) le calcul automatique des gradients (autograd), (2) le support GPU natif, (3) des modules de réseaux de neurones prêts à l'emploi. Godoy recommande de toujours commencer par comprendre les opérations NumPy avant de passer à PyTorch." },

            { type: "code", title: "PyTorch : Réseau superficiel", language: "python", description: "Voici comment vous définiriez ce réseau en PyTorch. Étudiez la structure puis calculez le nombre de paramètres.", initialCode: "# En PyTorch (pseudo-code, exécutez mentalement) :\n#\n# import torch\n# import torch.nn as nn\n#\n# model = nn.Sequential(\n#     nn.Linear(1, 3),   # couche cachée: 1 entrée → 3 unités\n#     nn.ReLU(),         # activation ReLU\n#     nn.Linear(3, 1),   # couche sortie: 3 → 1\n# )\n\n# Compter les paramètres :\n# nn.Linear(1, 3) : 1*3 poids + 3 biais = 6\n# nn.Linear(3, 1) : 3*1 poids + 1 biais = 4\n# Total : 10\n\nparams_hidden = 1 * 3 + 3\nparams_output = 3 * 1 + 1\ntotal = params_hidden + params_output\nprint(total)", expectedOutput: "10", solution: "# nn.Linear(in, out) crée une couche avec:\n#   - in * out poids\n#   - out biais\n#\n# Couche cachée nn.Linear(1, 3):\n#   1 × 3 = 3 poids + 3 biais = 6 params\n#\n# Couche sortie nn.Linear(3, 1):\n#   3 × 1 = 3 poids + 1 biais = 4 params\n#\n# Total: 6 + 4 = 10 paramètres\n# C'est exactement les 10 paramètres de l'équation 3.1!\n\nparams_hidden = 1 * 3 + 3  # 6\nparams_output = 3 * 1 + 1  # 4\ntotal = params_hidden + params_output  # 10\nprint(total)", hints: ["nn.Linear(in, out) a in×out poids et out biais", "Comptez séparément pour chaque couche"] },

            // ─── 3.8 Résumé ───
            { type: "heading", text: "3.8 Résumé" },
            { type: "paragraph", text: "Les réseaux de neurones superficiels ont une couche cachée. Ils (i) calculent plusieurs fonctions linéaires de l'entrée, (ii) passent chaque résultat par une fonction d'activation (typiquement ReLU), puis (iii) prennent une combinaison linéaire de ces activations pour former les sorties. Le théorème d'approximation universelle garantit qu'avec suffisamment d'unités, ce réseau peut approximer n'importe quelle fonction continue." },

            { type: "heading", text: "Problèmes" },
            { type: "callout", variant: "warning", title: "Problème 3.1", text: "Si la fonction d'activation était linéaire (a[z] = ψ₀ + ψ₁z), quel type de mapping serait créé ? Et si on supprimait l'activation (a[z] = z) ?" },
            { type: "callout", variant: "warning", title: "Problème 3.2", text: "Pour chacune des quatre régions linéaires, indiquez quelles unités cachées sont inactives (éteintes par ReLU) et lesquelles sont actives." },
            { type: "callout", variant: "warning", title: "Problème 3.3", text: "Dérivez des expressions pour les positions des 'joints' dans la fonction en termes des dix paramètres." },

            { type: "code", title: "Problème 3.1 — Activation linéaire", language: "python", description: "Si a[z] = z (activation identité), montrez que le réseau entier se réduit à une fonction linéaire de x. Calculez y pour x=2.", initialCode: "# Avec activation identité a[z] = z :\n# h_d = theta_d0 + theta_d1 * x (pas de ReLU)\n# y = phi_0 + phi_1*h_1 + phi_2*h_2 + phi_3*h_3\n#\n# Substituons :\n# y = phi_0 + phi_1*(t10 + t11*x) + phi_2*(t20 + t21*x) + phi_3*(t30 + t31*x)\n# y = [phi_0 + phi_1*t10 + phi_2*t20 + phi_3*t30] + [phi_1*t11 + phi_2*t21 + phi_3*t31]*x\n# C'est linéaire en x !\n\nphi_0 = 0.2; phi_1 = 1.0; phi_2 = -0.5; phi_3 = 0.8\nt10 = -1.0; t11 = 2.0\nt20 = 0.5;  t21 = -1.0\nt30 = -0.5; t31 = 0.5\n\nbias = phi_0 + phi_1*t10 + phi_2*t20 + phi_3*t30\nslope = phi_1*t11 + phi_2*t21 + phi_3*t31\n\nx = 2\ny = bias + slope * x\nprint(round(bias, 1))\nprint(round(slope, 1))\nprint(round(y, 1))", expectedOutput: "-1.5\n1.9\n2.3", hints: ["Substituez a[z] = z dans l'équation du réseau", "Regroupez les termes constants et les termes en x"] },

            { type: "callout", variant: "warning", title: "Problème 3.5", text: "Prouvez que ReLU[α·z] = α·ReLU[z] pour α ∈ ℝ⁺ (propriété d'homogénéité non négative)." },
        ]
    },
    "4": {
        id: 4,
        title: "Réseaux de Neurones Profonds",
        description: "Les réseaux profonds à couches multiples peuvent approximer des fonctions complexes beaucoup plus efficacement que les réseaux superficiels, grâce à la composition et l'abstraction hiérarchique.",
        estimatedTime: "150 min",
        objectives: [
            "Comprendre la composition de réseaux de neurones",
            "Maîtriser la notation matricielle pour les réseaux profonds",
            "Comparer l'efficacité des réseaux profonds vs superficiels en termes de régions linéaires",
            "Implémenter un réseau profond pas à pas",
            "Analyser le rôle de la profondeur, de la largeur, et des hyperparamètres",
        ],
        content: [
            { type: "paragraph", text: "Le chapitre précédent décrivait les réseaux superficiels avec une seule couche cachée. Ce chapitre introduit les réseaux profonds, qui ont plus d'une couche cachée. Avec les activations ReLU, les deux types décrivent des fonctions linéaires par morceaux. Cependant, les réseaux profonds créent exponentiellement plus de régions linéaires par paramètre, ce qui les rend bien plus expressifs pour un budget de paramètres donné." },

            // ─── 4.1 Composition ───
            { type: "heading", text: "4.1 Composition de réseaux de neurones" },
            { type: "paragraph", text: "L'idée fondamentale est simple : passer la sortie d'un réseau superficiel dans un second réseau superficiel. Le premier réseau 'plie' l'espace d'entrée, et le second applique une nouvelle transformation sur cet espace plié." },
            { type: "paragraph", text: "Considérons deux réseaux superficiels composés. Le premier prend x et produit y via 3 unités cachées :" },
            { type: "math-block", latex: "h_d = a[\\theta_{d0} + \\theta_{d1} x], \\quad y = \\phi_0 + \\sum_{d=1}^{3} \\phi_d h_d" },
            { type: "paragraph", text: "Le second prend y comme entrée et produit y' :" },
            { type: "math-block", latex: "h'_d = a[\\theta'_{d0} + \\theta'_{d1} y], \\quad y' = \\phi'_0 + \\sum_{d=1}^{3} \\phi'_d h'_d" },
            { type: "paragraph", text: "Le réseau composé peut créer des fonctions bien plus complexes. Intuitivement, le premier réseau crée 4 régions linéaires, et pour chaque paire de régions dans le second réseau, la composition peut 'replier' les segments du premier." },

            { type: "callout", variant: "info", title: "L'analogie du papier plié (Prince)", text: "Imaginez que le premier réseau trace une fonction sur une feuille de papier. La composition avec le second réseau équivaut à plier cette feuille : chaque segment du premier réseau est dupliqué dans des régions différentes, multipliant le nombre effectif de segments." },

            { type: "code", title: "Composition de deux réseaux", language: "python", description: "Composez deux réseaux simples. Le premier a 2 unités ReLU, le second aussi. Calculez la sortie pour x = 1.0.", initialCode: "# Réseau 1 : x → y\n# Unités cachées\nx = 1.0\n\nh1 = max(0, -0.5 + 1.5 * x)   # = max(0, 1.0) = 1.0\nh2 = max(0, 0.5 + -1.0 * x)   # = max(0, -0.5) = 0\ny_mid = 0.1 + 0.8 * h1 + -0.3 * h2\n\nprint(round(y_mid, 1))\n\n# Réseau 2 : y_mid → y_final\nh3 = max(0, -0.2 + 0.5 * y_mid)\nh4 = max(0, 0.3 + -0.8 * y_mid)\ny_final = 0.0 + 1.2 * h3 + 0.5 * h4\n\nprint(round(h3, 2))\nprint(round(h4, 2))\nprint(round(y_final, 3))", expectedOutput: "0.9\n0.25\n0\n0.3", hints: ["Calculez d'abord la sortie du premier réseau", "Utilisez cette sortie comme entrée du second réseau", "N'oubliez pas ReLU à chaque couche cachée"] },

            // ─── 4.2 Vers les réseaux profonds ───
            { type: "heading", text: "4.2 De la composition aux réseaux profonds" },
            { type: "paragraph", text: "La composition de deux réseaux est un cas particulier d'un réseau profond à deux couches cachées. Pour passer au cas général, nous éliminons la couche de sortie intermédiaire et connectons directement les unités cachées de la première couche aux unités cachées de la seconde." },

            // ─── 4.3 Cas général ───
            { type: "heading", text: "4.3 Réseaux de neurones profonds : cas général" },
            { type: "paragraph", text: "Un réseau profond avec K couches cachées calcule séquentiellement :" },
            { type: "math-block", latex: "\\mathbf{h}_1 = a[\\boldsymbol{\\beta}_0 + \\boldsymbol{\\Omega}_0 \\mathbf{x}]" },
            { type: "math-block", latex: "\\mathbf{h}_k = a[\\boldsymbol{\\beta}_{k-1} + \\boldsymbol{\\Omega}_{k-1} \\mathbf{h}_{k-1}] \\quad \\text{pour } k = 2, \\dots, K" },
            { type: "math-block", latex: "\\mathbf{y} = \\boldsymbol{\\beta}_K + \\boldsymbol{\\Omega}_K \\mathbf{h}_K" },
            { type: "paragraph", text: "Chaque couche k a ses propres paramètres : un vecteur de biais βₖ₋₁ et une matrice de poids Ωₖ₋₁. L'ensemble des paramètres est φ = {βₖ, Ωₖ} pour k = 0, ..., K." },

            { type: "interactive", component: "DeepNetworkDiagram" },

            { type: "code", title: "Forward pass d'un réseau profond (2 couches)", language: "python", description: "Implémentez le forward pass d'un réseau à 2 couches cachées de 3 unités chacune, pour une entrée scalaire x=0.5.", initialCode: "x = 0.5\n\n# === Couche 1 : x → h1 (3 unités) ===\nh1_1 = max(0, 0.3 + 1.0 * x)   # = max(0, 0.8)\nh1_2 = max(0, -0.5 + 2.0 * x)  # = max(0, 0.5)\nh1_3 = max(0, 0.1 + -1.5 * x)  # = max(0, -0.65)\n\nprint(h1_1)\nprint(h1_2)\nprint(h1_3)\n\n# === Couche 2 : h1 → h2 (3 unités) ===\nh2_1 = max(0, 0.1 + 0.5*h1_1 + -0.3*h1_2 + 0.8*h1_3)\nh2_2 = max(0, -0.2 + 0.7*h1_1 + 0.4*h1_2 + -0.1*h1_3)\nh2_3 = max(0, 0.0 + -0.5*h1_1 + 1.0*h1_2 + 0.3*h1_3)\n\nprint(round(h2_1, 2))\nprint(round(h2_2, 2))\nprint(round(h2_3, 2))\n\n# === Sortie : h2 → y ===\ny = 0.1 + 1.0*h2_1 + -0.5*h2_2 + 0.3*h2_3\nprint(round(y, 3))", expectedOutput: "0.8\n0.5\n0\n0.35\n0.56\n0.1\n0.25", hints: ["Procédez couche par couche, de gauche à droite", "Appliquez ReLU à chaque couche cachée", "La couche de sortie est linéaire (pas de ReLU)"] },

            // ─── 4.4 Hyperparamètres ───
            { type: "heading", text: "4.4 Hyperparamètres" },
            { type: "paragraph", text: "Un réseau profond est défini par plusieurs hyperparamètres structurels :" },
            { type: "list", items: [
                "Profondeur (K) : nombre de couches cachées. Plus le réseau est profond, plus il peut capturer des abstractions hiérarchiques.",
                "Largeur (D) : nombre d'unités dans chaque couche cachée. Plus la couche est large, plus elle a de capacité.",
                "Fonction d'activation : ReLU, Leaky ReLU, GELU, etc.",
                "Architecture : comment les couches sont connectées (feed-forward, skip connections, etc.).",
            ] },
            { type: "callout", variant: "tip", title: "Règle pratique (Géron)", text: "Pour la plupart des problèmes, commencez avec 2-5 couches cachées de 100-300 neurones chacune, avec ReLU. Augmentez la profondeur avant d'augmenter la largeur. Utilisez un learning rate scheduler et le early stopping." },

            { type: "interactive", component: "MatrixVis" },

            // ─── 4.5 Profond vs Superficiel ───
            { type: "heading", text: "4.5 Réseaux profonds vs superficiels" },
            { type: "paragraph", text: "Le résultat clé est que les réseaux profonds créent exponentiellement plus de régions linéaires que les réseaux superficiels pour le même nombre de paramètres :" },
            { type: "list", items: [
                "Réseau superficiel avec D unités : au plus D+1 régions linéaires.",
                "Réseau profond avec K couches de D unités : jusqu'à (D+1)^K régions linéaires (entrée 1D).",
                "Cela signifie qu'un réseau profond avec 10 couches de 10 unités peut créer jusqu'à 11^10 ≈ 2.6 × 10^10 régions, contre seulement 101 pour un réseau superficiel avec le même nombre total de paramètres !",
            ] },

            { type: "code", title: "Régions linéaires : profond vs superficiel", language: "python", description: "Comparez le nombre maximum de régions linéaires pour un réseau superficiel vs profond.", initialCode: "# Réseau superficiel : D unités → D+1 régions\nD_shallow = 100\nregions_shallow = D_shallow + 1\nprint(regions_shallow)\n\n# Réseau profond : K couches de D unités → (D+1)^K régions (1D)\nK = 10\nD_deep = 10  # même budget total ~100 params\nregions_deep = (D_deep + 1) ** K\nprint(regions_deep)\n\n# Ratio\nratio = regions_deep / regions_shallow\nprint(round(ratio, 0))", expectedOutput: "101\n25937424601\n256806184.0", solution: "# Budget ~100 paramètres\n# Superficiel: 100 unités → 101 régions\n# Profond: 10 couches × 10 unités → 11^10 ≈ 26 milliards!\n# Le réseau profond est ~257 millions de fois plus expressif\n\nD_shallow = 100\nregions_shallow = D_shallow + 1  # 101\nprint(regions_shallow)\n\nK = 10; D_deep = 10\nregions_deep = (D_deep + 1) ** K  # 11^10\nprint(regions_deep)\n\nratio = regions_deep / regions_shallow\nprint(round(ratio, 0))", hints: ["Superficiel: max D+1 régions", "Profond (1D): max (D+1)^K régions"] },

            // ─── 4.6 Notation matricielle ───
            { type: "heading", text: "4.6 Notation matricielle" },
            { type: "paragraph", text: "La notation matricielle est essentielle pour implémenter efficacement les réseaux profonds :" },
            { type: "math-block", latex: "\\mathbf{h}_k = a\\left[\\boldsymbol{\\beta}_{k-1} + \\boldsymbol{\\Omega}_{k-1} \\mathbf{h}_{k-1}\\right]" },
            { type: "paragraph", text: "En PyTorch (Godoy), cela se traduit directement par nn.Linear qui encapsule la multiplication matricielle et l'ajout du biais, suivi par nn.ReLU() pour l'activation." },

            { type: "code", title: "Compter les paramètres d'un réseau profond", language: "python", description: "Calculez le nombre total de paramètres pour un réseau avec Dᵢ=5 entrées, K=3 couches de D=20 unités, et Dₒ=3 sorties.", initialCode: "Di = 5   # entrées\nD = 20   # unités par couche cachée\nK = 3    # nombre de couches cachées\nDo = 3   # sorties\n\n# Couche 1 : Di → D\nlayer1 = D * (Di + 1)\n\n# Couches 2 à K : D → D\nhidden_layers = (K - 1) * D * (D + 1)\n\n# Couche sortie : D → Do\noutput_layer = Do * (D + 1)\n\ntotal = layer1 + hidden_layers + output_layer\n\nprint(layer1)\nprint(hidden_layers)\nprint(output_layer)\nprint(total)", expectedOutput: "120\n840\n63\n1023", solution: "# Couche 1: D × (Di + 1) = 20 × 6 = 120\n# Couche 2-3: 2 × D × (D + 1) = 2 × 20 × 21 = 840\n# Sortie: Do × (D + 1) = 3 × 21 = 63\n# Total: 120 + 840 + 63 = 1023\n\nDi=5; D=20; K=3; Do=3\nlayer1 = D*(Di+1)           # 120\nhidden_layers = (K-1)*D*(D+1)  # 840\noutput_layer = Do*(D+1)     # 63\ntotal = layer1+hidden_layers+output_layer  # 1023\nprint(layer1)\nprint(hidden_layers)\nprint(output_layer)\nprint(total)", hints: ["La première couche cachée a des dimensions différentes des suivantes", "Chaque couche cachée après la première connecte D unités à D unités"] },

            // ─── 4.7 PyTorch ───
            { type: "heading", text: "4.7 Implémentation en PyTorch" },
            { type: "paragraph", text: "En PyTorch (Godoy, ch. 4), un réseau profond se construit avec nn.Sequential. Chaque couche est un nn.Linear suivi d'un nn.ReLU, sauf la dernière couche qui est purement linéaire." },

            { type: "code", title: "Architecture PyTorch d'un réseau profond", language: "python", description: "Décrivez l'architecture d'un réseau profond en pseudo-PyTorch et comptez les paramètres.", initialCode: "# Architecture PyTorch (pseudo-code) :\n# model = nn.Sequential(\n#     nn.Linear(784, 256),  # MNIST: 28x28 → 256\n#     nn.ReLU(),\n#     nn.Linear(256, 128),  # 256 → 128\n#     nn.ReLU(),\n#     nn.Linear(128, 64),   # 128 → 64\n#     nn.ReLU(),\n#     nn.Linear(64, 10),    # 64 → 10 classes\n# )\n\n# Comptons les paramètres\np1 = 784 * 256 + 256\np2 = 256 * 128 + 128\np3 = 128 * 64 + 64\np4 = 64 * 10 + 10\n\ntotal = p1 + p2 + p3 + p4\nprint(p1)\nprint(p2)\nprint(p3)\nprint(p4)\nprint(total)", expectedOutput: "200960\n32896\n8256\n650\n242762", solution: "# Layer 1: 784×256 + 256 = 200,960\n# Layer 2: 256×128 + 128 = 32,896\n# Layer 3: 128×64 + 64 = 8,256\n# Layer 4: 64×10 + 10 = 650\n# Total: 242,762 paramètres (~243K)\n\np1 = 784*256 + 256  # 200,960\np2 = 256*128 + 128  # 32,896\np3 = 128*64 + 64    # 8,256\np4 = 64*10 + 10     # 650\ntotal = p1+p2+p3+p4 # 242,762\nprint(p1)\nprint(p2)\nprint(p3)\nprint(p4)\nprint(total)", hints: ["nn.Linear(in, out) a in×out + out paramètres"] },

            // ─── 4.8 Résumé ───
            { type: "heading", text: "4.8 Résumé" },
            { type: "paragraph", text: "Les réseaux profonds composent plusieurs couches de transformations non linéaires. Cette composition crée exponentiellement plus de régions linéaires par paramètre que les réseaux superficiels, ce qui les rend beaucoup plus efficaces en pratique. La notation matricielle simplifie la description et correspond directement aux implémentations PyTorch." },

            { type: "heading", text: "Problèmes" },
            { type: "callout", variant: "warning", title: "Problème 4.1", text: "Tracez la relation entre x et y' pour la composition de deux réseaux à 3 unités cachées chacun, pour x ∈ [-1, 1]. Combien de régions linéaires observez-vous ?" },
            { type: "callout", variant: "warning", title: "Problème 4.5", text: "Un réseau avec Dᵢ=5 entrées, Dₒ=1 sortie, K=20 couches de D=30 unités. Profondeur ? Largeur ? Nombre de paramètres ?" },

            { type: "code", title: "Problème 4.5 — Paramètres d'un grand réseau", language: "python", description: "Calculez les paramètres pour Dᵢ=5, Dₒ=1, K=20, D=30.", initialCode: "Di = 5; Do = 1; K = 20; D = 30\n\n# Couche 1: D × (Di + 1)\nlayer1 = D * (Di + 1)\n\n# Couches cachées 2 à K: (K-1) × D × (D + 1)\nhidden = (K - 1) * D * (D + 1)\n\n# Couche de sortie: Do × (D + 1)\noutput = Do * (D + 1)\n\ntotal = layer1 + hidden + output\n\nprint(f\"Profondeur: {K}\")\nprint(f\"Largeur: {D}\")\nprint(layer1)\nprint(hidden)\nprint(output)\nprint(total)", expectedOutput: "Profondeur: 20\nLargeur: 30\n180\n17670\n31\n17881", solution: "# Profondeur = K = 20 couches cachées\n# Largeur = D = 30 unités par couche\n# Couche 1: 30 × 6 = 180\n# Couches 2-20: 19 × 30 × 31 = 17,670\n# Sortie: 1 × 31 = 31\n# Total: 17,881\n\nDi=5; Do=1; K=20; D=30\nlayer1 = D*(Di+1)              # 180\nhidden = (K-1)*D*(D+1)         # 17670\noutput_l = Do*(D+1)            # 31\ntotal = layer1+hidden+output_l # 17881\nprint(f\"Profondeur: {K}\")\nprint(f\"Largeur: {D}\")\nprint(layer1)\nprint(hidden)\nprint(output_l)\nprint(total)", hints: ["Utilisez la formule générale des paramètres"] },

            { type: "callout", variant: "warning", title: "Problème 4.10", text: "Montrez qu'un réseau profond avec K couches de D unités a exactement D(Dᵢ+1) + (K-1)D(D+1) + Dₒ(D+1) paramètres." },
        ]
    }
};
