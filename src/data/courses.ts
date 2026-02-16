/* ═══════════════════════════════════════════
   Course Data Model & Sample Content
   ═══════════════════════════════════════════ */

export interface CourseNode {
  id: string;
  title: string;
  shortTitle: string;
  description: string;
  status: 'locked' | 'available' | 'in-progress' | 'completed';
  progress: number; // 0-100
  dependencies: string[];
  category: 'fundamentals' | 'architectures' | 'training' | 'advanced';
  exercises: Exercise[];
  theory: TheoryBlock[];
  codeTemplate: string;
  expectedOutput?: string;
}

export interface Exercise {
  id: string;
  title: string;
  instructions: string;
  starterCode: string;
  solution: string;
  hints: string[];
  completed: boolean;
}

export interface TheoryBlock {
  type: 'text' | 'equation' | 'diagram' | 'callout';
  content: string;
  label?: string;
  highlightVar?: string; // variable name that links to code
}

// ── Sample Course Data ──
// Content enriched from "Understanding Deep Learning" by Simon J.D. Prince (MIT Press)
export const courseNodes: CourseNode[] = [
  // ═══════════════════════════════════════
  // MODULE 1 — SUPERVISED LEARNING INTRO
  // ═══════════════════════════════════════
  {
    id: 'supervised-learning',
    title: 'Introduction à l\'Apprentissage Supervisé',
    shortTitle: 'Supervisé',
    description: 'Les bases de l\'apprentissage supervisé : modèles, paramètres, fonctions de perte et entraînement.',
    status: 'available',
    progress: 0,
    dependencies: [],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `En **apprentissage supervisé**, on construit un modèle **f[x, ϕ]** qui prend une entrée **x** et produit une prédiction **y**. Le modèle est une équation mathématique de forme fixe contenant des **paramètres ϕ** qui déterminent la relation entre entrée et sortie.`,
      },
      {
        type: 'equation',
        content: 'y = f[\\mathbf{x}, \\boldsymbol{\\phi}]',
        label: 'Modèle de prédiction',
        highlightVar: 'y',
      },
      {
        type: 'text',
        content: `**Entraîner** un modèle consiste à trouver les paramètres **ϕ** qui minimisent une **fonction de perte L[ϕ]**. Cette perte quantifie l'écart entre les prédictions du modèle et les sorties réelles du jeu de données d'entraînement {xᵢ, yᵢ}.`,
      },
      {
        type: 'equation',
        content: '\\hat{\\boldsymbol{\\phi}} = \\underset{\\boldsymbol{\\phi}}{\\text{argmin}} \\; \\mathcal{L}[\\boldsymbol{\\phi}]',
        label: 'Minimisation de la perte',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `L'exemple le plus simple est la **régression linéaire 1D** : le modèle décrit une droite y = ϕ₀ + ϕ₁x, où ϕ₀ est l'ordonnée à l'origine et ϕ₁ la pente. La perte des **moindres carrés** mesure la somme des carrés des écarts :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}[\\boldsymbol{\\phi}] = \\sum_{i=1}^{I} (f[x_i, \\boldsymbol{\\phi}] - y_i)^2',
        label: 'Perte des moindres carrés (Least Squares)',
        highlightVar: 'loss',
      },
      {
        type: 'callout',
        content: '💡 Après l\'entraînement, on évalue le modèle sur des **données de test** séparées pour mesurer sa capacité de **généralisation** — c\'est-à-dire sa performance sur des exemples qu\'il n\'a jamais vus.',
      },
    ],
    exercises: [
      {
        id: 'sl-ex1',
        title: 'Régression linéaire depuis zéro',
        instructions: 'Implémentez une régression linéaire simple y = ϕ₀ + ϕ₁x et calculez la perte des moindres carrés.',
        starterCode: `import torch

# Données d'entraînement
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([2.1, 3.9, 6.2, 7.8, 10.1])

# Paramètres du modèle (à initialiser)
phi_0 = ___  # ordonnée à l'origine
phi_1 = ___  # pente

# Prédiction : y = phi_0 + phi_1 * x
y_pred = ___

# Perte des moindres carrés
loss = ___

print(f"Prédictions: {y_pred}")
print(f"Perte: {loss.item():.4f}")`,
        solution: `import torch

# Données d'entraînement
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([2.1, 3.9, 6.2, 7.8, 10.1])

# Paramètres du modèle
phi_0 = torch.tensor(0.0)  # ordonnée à l'origine
phi_1 = torch.tensor(2.0)  # pente

# Prédiction : y = phi_0 + phi_1 * x
y_pred = phi_0 + phi_1 * x

# Perte des moindres carrés
loss = torch.sum((y_pred - y) ** 2)

print(f"Prédictions: {y_pred}")
print(f"Perte: {loss.item():.4f}")`,
        hints: [
          'Initialisez phi_0 et phi_1 avec torch.tensor(valeur)',
          'y_pred = phi_0 + phi_1 * x',
          'loss = torch.sum((y_pred - y) ** 2)',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch

# ══ Apprentissage Supervisé — Régression Linéaire ══

# Données d'entraînement (entrée → sortie)
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = torch.tensor([2.1, 3.9, 6.2, 7.8, 10.1])

# Modèle : y = phi_0 + phi_1 * x
phi_0 = torch.tensor(0.0, requires_grad=True)
phi_1 = torch.tensor(0.0, requires_grad=True)

# Entraînement par descente de gradient
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    y_pred = phi_0 + phi_1 * x_train
    
    # Perte des moindres carrés
    loss = torch.sum((y_pred - y_train) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Mise à jour des paramètres
    with torch.no_grad():
        phi_0 -= learning_rate * phi_0.grad
        phi_1 -= learning_rate * phi_1.grad
        phi_0.grad.zero_()
        phi_1.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, y={phi_1.item():.2f}x + {phi_0.item():.2f}")

print(f"\\nModèle final: y = {phi_1.item():.2f}x + {phi_0.item():.2f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 2 — TENSEURS
  // ═══════════════════════════════════════
  {
    id: 'tensors',
    title: 'Tenseurs & Opérations',
    shortTitle: 'Tenseurs',
    description: 'Comprendre les tenseurs — la structure de données fondamentale du Deep Learning — et leurs opérations.',
    status: 'available',
    progress: 0,
    dependencies: ['supervised-learning'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Un **tenseur** est la structure de données fondamentale du Deep Learning. C'est une généralisation des matrices à N dimensions. En Deep Learning, toutes les données (images, texte, audio) et tous les paramètres des modèles sont représentés comme des tenseurs.\n\n- **Scalaire** : tenseur de rang 0 (un seul nombre)\n- **Vecteur** : tenseur de rang 1 (liste de nombres)\n- **Matrice** : tenseur de rang 2 (grille 2D)\n- **Tenseur 3D+** : rang 3+ (ex: image RGB = 3×H×W)`,
      },
      {
        type: 'equation',
        content: '\\mathbf{T} \\in \\mathbb{R}^{d_1 \\times d_2 \\times \\cdots \\times d_n}',
        label: 'Forme d\'un tenseur de rang n',
      },
      {
        type: 'callout',
        content: '💡 En PyTorch, `torch.tensor()` crée un tenseur depuis des données Python. Utilisez `.shape` pour inspecter ses dimensions et `.dtype` pour son type.',
      },
      {
        type: 'text',
        content: `La **multiplication matricielle** est l'opération la plus fondamentale en Deep Learning. C'est elle qui implémente les transformations linéaires au cœur de chaque couche neuronale. Pour que A×B soit défini, le nombre de colonnes de A doit égaler le nombre de lignes de B.`,
      },
      {
        type: 'equation',
        content: '\\mathbf{C} = \\mathbf{A} \\mathbf{B} \\quad \\text{où } C_{ij} = \\sum_{k} A_{ik} B_{kj}',
        label: 'Multiplication matricielle',
        highlightVar: 'result',
      },
      {
        type: 'text',
        content: `Le **broadcasting** permet d'effectuer des opérations entre tenseurs de tailles différentes. PyTorch aligne automatiquement les dimensions en partant de la droite et expand les dimensions de taille 1.`,
      },
    ],
    exercises: [
      {
        id: 'tensors-ex1',
        title: 'Créer et manipuler des tenseurs',
        instructions: 'Créez un tenseur 3×4 rempli de uns, multipliez-le par un tenseur 4×2 aléatoire, et affichez les formes.',
        starterCode: `import torch

# Créez un tenseur 3x4 rempli de uns
A = ___

# Créez un tenseur 4x2 aléatoire (normal)
B = ___

# Multiplication matricielle
result = ___

print(f"Shape de A: {A.shape}")
print(f"Shape de B: {B.shape}")
print(f"Shape du résultat: {result.shape}")
print(f"Résultat:\\n{result}")`,
        solution: `import torch

# Créez un tenseur 3x4 rempli de uns
A = torch.ones(3, 4)

# Créez un tenseur 4x2 aléatoire (normal)
B = torch.randn(4, 2)

# Multiplication matricielle
result = torch.matmul(A, B)

print(f"Shape de A: {A.shape}")
print(f"Shape de B: {B.shape}")
print(f"Shape du résultat: {result.shape}")
print(f"Résultat:\\n{result}")`,
        hints: [
          'torch.ones(rows, cols) crée un tenseur de uns',
          'torch.randn(rows, cols) génère des valeurs aléatoires normales',
          'torch.matmul(A, B) ou A @ B pour la multiplication matricielle',
        ],
        completed: false,
      },
      {
        id: 'tensors-ex2',
        title: 'Broadcasting et opérations',
        instructions: 'Créez un tenseur 3×1 et un tenseur 1×4. Additionnez-les pour obtenir un tenseur 3×4 grâce au broadcasting.',
        starterCode: `import torch

# Tenseur colonne 3x1
col = ___

# Tenseur ligne 1x4
row = ___

# Broadcasting : 3x1 + 1x4 → 3x4
result = ___

print(f"col shape: {col.shape}")
print(f"row shape: {row.shape}")
print(f"result shape: {result.shape}")
print(f"result:\\n{result}")`,
        solution: `import torch

# Tenseur colonne 3x1
col = torch.tensor([[1.0], [2.0], [3.0]])

# Tenseur ligne 1x4
row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

# Broadcasting : 3x1 + 1x4 → 3x4
result = col + row

print(f"col shape: {col.shape}")
print(f"row shape: {row.shape}")
print(f"result shape: {result.shape}")
print(f"result:\\n{result}")`,
        hints: [
          'Utilisez torch.tensor([[1.0], [2.0], [3.0]]) pour une colonne 3x1',
          'Le broadcasting aligne les dimensions depuis la droite',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch

# ══ Exploration des Tenseurs ══

# Scalaire (rang 0)
scalar = torch.tensor(42.0)
print(f"Scalaire: {scalar}, shape: {scalar.shape}, ndim: {scalar.ndim}")

# Vecteur (rang 1)
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vecteur: {vector}, shape: {vector.shape}")

# Matrice (rang 2)
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
print(f"Matrice shape: {matrix.shape}")

# Tenseur 3D (rang 3) — comme une image RGB
image = torch.randn(3, 224, 224)
print(f"Image tensor shape: {image.shape}")

# ── Opérations fondamentales ──
A = torch.ones(3, 4)
B = torch.randn(4, 2)
result = A @ B  # multiplication matricielle
print(f"\\n{A.shape} × {B.shape} = {result.shape}")

# Broadcasting
col = torch.tensor([[1.0], [2.0], [3.0]])  # 3x1
row = torch.tensor([[10.0, 20.0, 30.0]])    # 1x3
print(f"\\nBroadcasting: {col.shape} + {row.shape} = {(col + row).shape}")
print(col + row)
`,
  },

  // ═══════════════════════════════════════
  // MODULE 3 — RÉSEAUX SUPERFICIELS
  // (Ch. 3 — Understanding Deep Learning)
  // ═══════════════════════════════════════
  {
    id: 'shallow-networks',
    title: 'Réseaux de Neurones Superficiels',
    shortTitle: 'Shallow Net',
    description: 'Comprendre les réseaux à une couche cachée, ReLU, les régions linéaires et le théorème d\'approximation universelle (Ch. 3 — UDL).',
    status: 'available',
    progress: 0,
    dependencies: ['tensors'],
    category: 'fundamentals',
    theory: [
      // ──────── SECTION 1 : INTRODUCTION ────────
      {
        type: 'text',
        content: `Le chapitre 2 a introduit la régression linéaire (une droite). Mais une droite ne peut pas capturer de relations complexes. Les **réseaux superficiels** (shallow neural networks) décrivent des **fonctions linéaires par morceaux** (piecewise linear functions) suffisamment expressives pour approximer n'importe quelle relation entre entrées et sorties.`,
      },
      {
        type: 'callout',
        content: '💡 Un réseau "superficiel" désigne un réseau avec **une seule couche cachée** (hidden layer). Ce terme contraste avec "profond" (deep), qui désigne les réseaux à plusieurs couches cachées (Ch. 4).',
      },

      // ──────── SECTION 2 : L'EXEMPLE DU RÉSEAU ────────
      {
        type: 'text',
        content: `## 3.1 — Exemple de réseau neuronal\n\nConsidérons un réseau avec **10 paramètres** ϕ = {ϕ₀, ϕ₁, ϕ₂, ϕ₃, θ₁₀, θ₁₁, θ₂₀, θ₂₁, θ₃₀, θ₃₁} qui transforme un scalaire x en un scalaire y. Le calcul se fait en **3 étapes** :`,
      },
      {
        type: 'diagram',
        content: `      ÉTAPE 1             ÉTAPE 2              ÉTAPE 3
  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐
  │ 3 fonctions │   │  Activation  │   │   Combinaison     │
  │ linéaires   │──▶│   ReLU a[•]  │──▶│   linéaire        │
  │ de l'entrée │   │  (clip < 0)  │   │   + offset ϕ₀     │
  └─────────────┘   └──────────────┘   └───────────────────┘

  θ₁₀ + θ₁₁·x  ──▶  h₁ = ReLU[•] ──┐
                                      ├──▶ y = ϕ₀ + ϕ₁h₁ + ϕ₂h₂ + ϕ₃h₃
  θ₂₀ + θ₂₁·x  ──▶  h₂ = ReLU[•] ──┤
                                      │
  θ₃₀ + θ₃₁·x  ──▶  h₃ = ReLU[•] ──┘`,
        label: 'Fig. 3.3 — Pipeline de calcul d\'un réseau superficiel',
      },
      {
        type: 'equation',
        content: 'y = f[x, \\boldsymbol{\\phi}] = \\phi_0 + \\phi_1 \\, a[\\theta_{10} + \\theta_{11} x] + \\phi_2 \\, a[\\theta_{20} + \\theta_{21} x] + \\phi_3 \\, a[\\theta_{30} + \\theta_{31} x]',
        label: 'Éq. 3.1 — Réseau superficiel (Shallow Network)',
        highlightVar: 'output',
      },

      // ──────── SECTION 3 : ReLU ────────
      {
        type: 'text',
        content: `## 3.1.1 — La fonction d'activation ReLU\n\nLa fonction d'activation **a[•]** est ce qui rend le réseau **non-linéaire**. Sans elle, le réseau ne serait qu'une autre fonction linéaire (voir Problème 3.1). Le choix le plus courant est le **ReLU** (Rectified Linear Unit) :`,
      },
      {
        type: 'equation',
        content: 'a[z] = \\text{ReLU}[z] = \\max(0, z) = \\begin{cases} 0 & \\text{si } z < 0 \\\\ z & \\text{si } z \\geq 0 \\end{cases}',
        label: 'Éq. 3.2 — Rectified Linear Unit (ReLU)',
        highlightVar: 'relu',
      },
      {
        type: 'diagram',
        content: `  y ▲
    │        ╱
    │       ╱
    │      ╱   ← pente = 1
    │     ╱
    │    ╱
  ──┼───╱──────────▶ z
    │  ╱
    │ (clipped à 0 pour z < 0)
    │`,
        label: 'Fig. 3.1 — Graphe du ReLU : retourne z si z ≥ 0, sinon 0',
      },
      {
        type: 'text',
        content: `**En PyTorch**, le ReLU existe sous 3 formes :\n\n- **\`torch.relu(tensor)\`** — fonction simple appliquée à un tenseur\n- **\`torch.clamp(tensor, min=0)\`** — équivalent plus explicite\n- **\`nn.ReLU()\`** — module réutilisable dans un \`nn.Sequential\` ou \`nn.Module\`\n\n**Pourquoi ReLU est si populaire ?** Sa dérivée vaut 1 pour z > 0 et 0 pour z < 0. Cela rend le gradient stable pendant la backpropagation (contrairement au sigmoid/tanh qui "saturent").`,
      },
      {
        type: 'callout',
        content: '⚠ **Le problème du "dying ReLU"** : si toutes les entrées d\'un neurone sont négatives, le ReLU retourne toujours 0 et le gradient est nul. Ce neurone est "mort" — il ne peut plus apprendre. Solutions : Leaky ReLU (pente 0.01 pour z < 0), Parametric ReLU, ou ELU.',
      },

      // ──────── SECTION 4 : UNITÉS CACHÉES ────────
      {
        type: 'text',
        content: `## 3.1.2 — Les unités cachées (Hidden Units)\n\nLe calcul se décompose naturellement en **unités cachées** h₁, h₂, h₃. Chaque unité est un neurone qui applique une fonction linéaire de l'entrée puis la passe par ReLU :`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} h_1 &= a[\\theta_{10} + \\theta_{11}x] \\\\ h_2 &= a[\\theta_{20} + \\theta_{21}x] \\\\ h_3 &= a[\\theta_{30} + \\theta_{31}x] \\end{aligned}',
        label: 'Éq. 3.3 — Calcul des unités cachées',
        highlightVar: 'relu',
      },
      {
        type: 'text',
        content: `Puis la sortie combine linéairement ces unités cachées :`,
      },
      {
        type: 'equation',
        content: 'y = \\phi_0 + \\phi_1 h_1 + \\phi_2 h_2 + \\phi_3 h_3',
        label: 'Éq. 3.4 — Combinaison linéaire de sortie',
        highlightVar: 'output',
      },

      // ──────── SECTION 5 : PATTERNS D'ACTIVATION ────────
      {
        type: 'text',
        content: `## 3.1.3 — Régions linéaires & Patterns d'activation\n\nChaque unité cachée crée un **"joint"** (coude) dans la fonction de sortie — le point où la droite θ•₀ + θ•₁x croise zéro. De part et d'autre de ce joint, l'unité est soit **active** (z ≥ 0, passe l'entrée) soit **inactive** (z < 0, retourne 0).\n\nAvec 3 unités cachées, on obtient jusqu'à **4 régions linéaires** et **3 joints**. Chaque région correspond à un **pattern d'activation** différent :`,
      },
      {
        type: 'diagram',
        content: `  y ▲
    │         ╱╲
    │        ╱  ╲          ╱
    │       ╱    ╲        ╱
    │      ╱      ╲      ╱
    │     ╱        ╲    ╱
    │    ╱    R2    ╲  ╱
    │   ╱            ╲╱
    │  ╱  R1      R3    R4
  ──┼─╱──────┼──────┼──────┼──▶ x
    │  joint₁   joint₂   joint₃

  R1 : h₁=off, h₂=off, h₃=off  →  pente = 0
  R2 : h₁=on,  h₂=off, h₃=off  →  pente = ϕ₁·θ₁₁
  R3 : h₁=on,  h₂=on,  h₃=off  →  pente = ϕ₁·θ₁₁ + ϕ₂·θ₂₁
  R4 : h₁=on,  h₂=on,  h₃=on   →  pente = ϕ₁·θ₁₁ + ϕ₂·θ₂₁ + ϕ₃·θ₃₁`,
        label: 'Fig. 3.2 — Fonction linéaire par morceaux avec 4 régions',
      },
      {
        type: 'callout',
        content: '🧠 **Intuition clé** : la pente de chaque région est la somme des pentes θ•₁ × ϕ• des unités **actives** dans cette région. L\'offset ϕ₀ contrôle la hauteur globale. C\'est ainsi qu\'on "dessine" des fonctions complexes morceau par morceau.',
      },

      // ──────── SECTION 6 : NOTATION MATRICIELLE ────────
      {
        type: 'text',
        content: `## 3.2 — Notation matricielle & PyTorch\n\nOn regroupe le calcul en notation matricielle. Soit **β₀** le vecteur de biais de la couche cachée, **Ω₀** la matrice de poids d'entrée, **β₁** le biais de sortie, et **ω₁** les poids de sortie :`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h} = a\\!\\left[\\boldsymbol{\\beta}_0 + \\boldsymbol{\\Omega}_0 \\mathbf{x}\\right] \\qquad y = \\beta_1 + \\boldsymbol{\\omega}_1^T \\mathbf{h}',
        label: 'Notation matricielle compacte',
        highlightVar: 'hidden',
      },
      {
        type: 'text',
        content: `**En PyTorch, \`nn.Linear(in, out)\`** implémente exactement cette opération :\n\n- Il stocke une matrice de poids **W** de taille (out × in)\n- Un vecteur de biais **b** de taille (out)\n- La sortie est : **output = x @ W.T + b**\n\nUn réseau superficiel complet se construit avec :`,
      },
      {
        type: 'diagram',
        content: `  ┌──────────────────────────────────────────────────┐
  │  model = nn.Sequential(                          │
  │      nn.Linear(D_i, D),    # Ω₀·x + β₀          │
  │      nn.ReLU(),            # a[•]                │
  │      nn.Linear(D, D_o)     # ω₁ᵀ·h + β₁          │
  │  )                                               │
  └──────────────────────────────────────────────────┘

  Où :
    D_i = dimension d'entrée
    D   = nombre d'unités cachées (hidden units)
    D_o = dimension de sortie`,
        label: 'Construction PyTorch d\'un réseau superficiel',
      },
      {
        type: 'text',
        content: `**Fonctions PyTorch utiles pour cette étape :**\n\n- **\`nn.Sequential(*layers)\`** : empile des couches en séquence, forward automatique\n- **\`nn.Linear(in_features, out_features)\`** : couche dense (transformation affine)\n- **\`nn.ReLU()\`** : module d'activation ReLU (réutilisable)\n- **\`model.parameters()\`** : itérateur sur tous les poids/biais du modèle\n- **\`sum(p.numel() for p in model.parameters())\`** : compte le nombre total de paramètres`,
      },

      // ──────── SECTION 7 : NOMBRE DE PARAMÈTRES ────────
      {
        type: 'text',
        content: `## 3.3 — Comptage des paramètres\n\nUn réseau superficiel avec **Dᵢ** entrées, **D** unités cachées et **Dₒ** sorties a :`,
      },
      {
        type: 'equation',
        content: 'N_{\\text{params}} = \\underbrace{D \\cdot (D_i + 1)}_{\\text{couche cachée}} + \\underbrace{D_o \\cdot (D + 1)}_{\\text{couche de sortie}}',
        label: 'Éq. — Nombre de paramètres (Problème 3.17)',
      },
      {
        type: 'text',
        content: `Exemple : Dᵢ = 1, D = 3, Dₒ = 1 → 3×(1+1) + 1×(3+1) = 6 + 4 = **10 paramètres** — exactement les 10 de l'Éq. 3.1 !\n\nExemple 2 : Dᵢ = 784 (image 28×28), D = 100, Dₒ = 10 → 100×785 + 10×101 = 78,500 + 1,010 = **79,510 paramètres**.`,
      },

      // ──────── SECTION 8 : THÉORÈME D'APPROXIMATION ────────
      {
        type: 'text',
        content: `## 3.4 — Théorème d'approximation universelle\n\nAvec D unités cachées et ReLU, le réseau crée au maximum **D + 1 régions linéaires**. Plus la fonction cible est complexe, plus il faut de régions (et donc d'unités) pour l'approximer :`,
      },
      {
        type: 'diagram',
        content: `  D = 2           D = 5               D = 20
  ▲               ▲                   ▲
  │ ╱╲            │    ╱╲              │  ·∼∼∼∼·
  │╱  ╲  ╱       │   ╱  ╲  ╱╲        │ ∫   f(x) dx
  │    ╲╱         │  ╱    ╲╱  ╲ ╱    │ ≈ somme de
  │               │ ╱          ╲╱     │   segments
  └──────▶       └──────────▶       └──────────▶
  3 régions       6 régions           ≈ courbe lisse`,
        label: 'Fig. 3.5 — Approximation : plus de hidden units → plus de régions → meilleure fidélité',
      },
      {
        type: 'callout',
        content: '🧠 **Théorème d\'approximation universelle** (Cybenko 1989, Hornik 1991) : pour toute fonction continue f définie sur un compact et tout ε > 0, il existe un réseau superficiel avec suffisamment d\'unités cachées tel que |f(x) - réseau(x)| < ε pour tout x. En d\'autres termes, un réseau à **une seule couche cachée peut approximer n\'importe quelle fonction continue** !',
      },

      // ──────── SECTION 9 : ENTRÉES/SORTIES MULTIDIMENSIONNELLES ────────
      {
        type: 'text',
        content: `## 3.5 — Entrées et sorties multidimensionnelles\n\n**Entrées multiples (Dᵢ > 1)** : chaque unité cachée reçoit une combinaison linéaire de **toutes** les entrées. Par exemple avec 2 entrées x = [x₁, x₂]ᵀ :`,
      },
      {
        type: 'equation',
        content: 'h_d = a\\!\\left[\\theta_{d0} + \\theta_{d1} x_1 + \\theta_{d2} x_2\\right]',
        label: 'Éq. 3.9 — Unité cachée avec 2 entrées',
      },
      {
        type: 'text',
        content: `En 2D, le ReLU crée des **hyperplans** (droites) qui divisent le plan d'entrée en **régions convexes polygonales**. Chaque région a une surface linéaire différente.\n\n**Sorties multiples (Dₒ > 1)** : on utilise une combinaison linéaire **différente** des mêmes unités cachées pour chaque sortie. Les joints restent aux mêmes positions, mais les pentes varient.`,
      },
      {
        type: 'text',
        content: `**Nombre de régions en haute dimension** :\n\nAvec Dᵢ ≥ 2 et D unités cachées, le nombre maximum de régions est donné par la formule de Zaslavsky (1975) :`,
      },
      {
        type: 'equation',
        content: 'N_{\\text{regions}} = \\sum_{j=0}^{D_i} \\binom{D}{j}',
        label: 'Formule de Zaslavsky — Nombre max de régions',
      },
      {
        type: 'text',
        content: `Avec Dᵢ dimensions et D ≥ Dᵢ unités cachées, on crée au minimum **2^Dᵢ** régions (en alignant chaque hyperplan avec un axe de coordonnées). Exemple : D = 500, Dᵢ = 100 → plus de **10¹⁰⁷** régions !`,
      },

      // ──────── SECTION 10 : TERMINOLOGIE ────────
      {
        type: 'text',
        content: `## 3.6 — Terminologie\n\nLe réseau est décrit en **couches** (layers) :\n\n- **Input layer** (couche d'entrée) : les données x\n- **Hidden layer** (couche cachée) : les neurones hd avec ReLU\n- **Output layer** (couche de sortie) : la prédiction y\n\nAutres termes importants :`,
      },
      {
        type: 'diagram',
        content: `  ╔═══════════════════════════════════════════════════════════╗
  ║                    VOCABULAIRE                            ║
  ╠══════════════════╤════════════════════════════════════════╣
  ║ Multi-layer      │ Tout réseau avec ≥ 1 couche cachée    ║
  ║ perceptron (MLP) │ (terme historique)                    ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Neurone / Unit   │ Un élément de la couche cachée        ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Pré-activation   │ Valeur AVANT le ReLU : θ₀ + θ₁x      ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Activation       │ Valeur APRÈS le ReLU : a[θ₀ + θ₁x]   ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Weights (poids)  │ Paramètres de pente (θ₁₁, ϕ₁, …)     ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Biases (biais)   │ Paramètres d'offset (θ₁₀, ϕ₀, …)     ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Feed-forward     │ Graphe acyclique (pas de boucles)     ║
  ╟──────────────────┼────────────────────────────────────────╢
  ║ Fully connected  │ Chaque neurone connecté à tous les    ║
  ║                  │ neurones de la couche suivante        ║
  ╚══════════════════╧════════════════════════════════════════╝`,
        label: 'Fig. 3.12 — Terminologie des réseaux de neurones',
      },

      // ──────── SECTION 11 : AUTRES ACTIVATIONS ────────
      {
        type: 'text',
        content: `## 3.7 — Autres fonctions d'activation\n\nLe ReLU n'est pas la seule option. Voici les alternatives les plus importantes :`,
      },
      {
        type: 'diagram',
        content: `  ┌─────────────────┬───────────────────────────────────────┐
  │ Activation      │ Formule                               │
  ├─────────────────┼───────────────────────────────────────┤
  │ Sigmoid σ(z)    │  1 / (1 + e⁻ᶻ)          ∈ (0, 1)     │
  │ Tanh            │  (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)  ∈ (-1, 1)   │
  │ Leaky ReLU      │  max(0.01z, z)                        │
  │ Parametric ReLU │  max(αz, z)   α appris                │
  │ ELU             │  z si z≥0, α(eᶻ-1) sinon             │
  │ Swish / SiLU    │  z · σ(βz)    β appris                │
  │ GELU            │  z · Φ(z)     Φ = CDF gaussienne      │
  │ Softplus        │  log(1 + eᶻ)  version lisse du ReLU   │
  └─────────────────┴───────────────────────────────────────┘`,
        label: 'Fig. 3.13 — Catalogue des fonctions d\'activation',
      },
      {
        type: 'text',
        content: `**En PyTorch**, chaque activation est disponible en module :\n\n- \`nn.ReLU()\`, \`nn.LeakyReLU(0.01)\`, \`nn.ELU(alpha=1.0)\`\n- \`nn.Sigmoid()\`, \`nn.Tanh()\`, \`nn.SiLU()\` (= Swish)\n- \`nn.GELU()\`, \`nn.Softplus()\`\n- \`nn.PReLU()\` — le paramètre α est appris pendant l'entraînement`,
      },

      // ──────── SECTION 12 : RÉSUMÉ ────────
      {
        type: 'callout',
        content: '⚡ **Résumé du Chapitre 3** :\n(1) Les réseaux superficiels calculent des fonctions linéaires par morceaux\n(2) Chaque unité cachée ajoute un "joint" et une région linéaire\n(3) Avec assez d\'unités, on approxime n\'importe quelle fonction continue\n(4) Le ReLU est l\'activation standard car son gradient est simple et stable\n(5) Le nombre de paramètres est D·(Dᵢ+1) + Dₒ·(D+1)',
      },
    ],
    exercises: [
      // ═══════════════════════════════════════
      // EXERCICE 1 — THÉORIQUE : Activation linéaire
      // ═══════════════════════════════════════
      {
        id: 'shallow-th1',
        title: '🧠 Théorie — Activation linéaire (Prob. 3.1)',
        instructions: 'Problème 3.1 du livre : que se passe-t-il si la fonction d\'activation est **linéaire** a[z] = ψ₀ + ψ₁z au lieu de ReLU ? Prouvez-le en code : créez un réseau avec activation linéaire et montrez que le résultat est toujours une simple droite (fonction affine), quel que soit le nombre d\'unités cachées.',
        starterCode: `import torch

# Activation linéaire : a[z] = psi_0 + psi_1 * z
psi_0, psi_1 = 0.5, 2.0

def linear_activation(z):
    return psi_0 + psi_1 * z

x = torch.tensor(1.5)

# Paramètres couche cachée
theta = torch.tensor([[0.5, -1.0],
                       [-0.3, 0.8],
                       [0.1, 1.2]])

# Paramètres sortie
phi = torch.tensor([0.2, 0.5, -0.3, 0.7])

# Calculez h1, h2, h3 avec activation LINÉAIRE
h1 = ___
h2 = ___
h3 = ___

# Sortie
y = ___

print(f"h1={h1.item():.4f}, h2={h2.item():.4f}, h3={h3.item():.4f}")
print(f"y = {y.item():.4f}")

# Maintenant montrez que y = A*x + B (constantes)
# Calculez A et B théoriquement
A = psi_1 * (phi[1]*theta[0,1] + phi[2]*theta[1,1] + phi[3]*theta[2,1])
B_offset = psi_0 * (phi[1] + phi[2] + phi[3])
B_theta  = psi_1 * (phi[1]*theta[0,0] + phi[2]*theta[1,0] + phi[3]*theta[2,0])
B = phi[0] + B_offset + B_theta

print(f"\\nVérification : y = {A:.4f} * x + {B:.4f}")
print(f"Calcul direct : {A * 1.5 + B:.4f}")
print(f"Conclusion : avec activation linéaire, le réseau est juste une DROITE !")`,
        solution: `import torch

psi_0, psi_1 = 0.5, 2.0

def linear_activation(z):
    return psi_0 + psi_1 * z

x = torch.tensor(1.5)

theta = torch.tensor([[0.5, -1.0],
                       [-0.3, 0.8],
                       [0.1, 1.2]])

phi = torch.tensor([0.2, 0.5, -0.3, 0.7])

h1 = linear_activation(theta[0, 0] + theta[0, 1] * x)
h2 = linear_activation(theta[1, 0] + theta[1, 1] * x)
h3 = linear_activation(theta[2, 0] + theta[2, 1] * x)

y = phi[0] + phi[1] * h1 + phi[2] * h2 + phi[3] * h3

print(f"h1={h1.item():.4f}, h2={h2.item():.4f}, h3={h3.item():.4f}")
print(f"y = {y.item():.4f}")

A = psi_1 * (phi[1]*theta[0,1] + phi[2]*theta[1,1] + phi[3]*theta[2,1])
B_offset = psi_0 * (phi[1] + phi[2] + phi[3])
B_theta  = psi_1 * (phi[1]*theta[0,0] + phi[2]*theta[1,0] + phi[3]*theta[2,0])
B = phi[0] + B_offset + B_theta

print(f"\\nVérification : y = {A:.4f} * x + {B:.4f}")
print(f"Calcul direct : {A * 1.5 + B:.4f}")
print(f"Conclusion : avec activation linéaire, le réseau est juste une DROITE !")`,
        hints: [
          'h1 = linear_activation(theta[0,0] + theta[0,1] * x)',
          'y = phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3',
          'Le résultat est toujours de la forme A·x + B, donc une droite !',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 2 — PRATIQUE : Forward pass ReLU
      // ═══════════════════════════════════════
      {
        id: 'shallow-pr1',
        title: '💻 Pratique — Forward pass avec ReLU',
        instructions: 'Implémentez un réseau superficiel avec 3 unités cachées et activation ReLU. Calculez la sortie pour x = -1.0, 0.0, 0.5, 1.5 et identifiez les patterns d\'activation (quelles unités sont actives/inactives à chaque x).',
        starterCode: `import torch

def relu(z):
    """ReLU : max(0, z)"""
    return ___

# Paramètres (tirés de la Figure 3.2a du livre)
theta = torch.tensor([[-0.2, 0.4],   # θ₁₀, θ₁₁
                       [-0.9, 0.9],   # θ₂₀, θ₂₁
                       [ 1.1, -0.7]]) # θ₃₀, θ₃₁

phi = torch.tensor([-0.23, -1.3, 1.3, 0.66])  # ϕ₀, ϕ₁, ϕ₂, ϕ₃

def shallow_forward(x, theta, phi):
    """Forward pass d'un réseau superficiel"""
    # Pré-activations
    z1 = ___
    z2 = ___
    z3 = ___
    
    # Activations (unités cachées)
    h1 = relu(z1)
    h2 = relu(z2)
    h3 = relu(z3)
    
    # Sortie
    y = ___
    
    # Pattern d'activation : 1 si actif, 0 si inactif
    pattern = (f"h1={'ON' if h1 > 0 else 'off'}, "
               f"h2={'ON' if h2 > 0 else 'off'}, "
               f"h3={'ON' if h3 > 0 else 'off'}")
    
    return y.item(), pattern

# Test sur plusieurs entrées
for x_val in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.5]:
    x = torch.tensor(x_val)
    y, pattern = shallow_forward(x, theta, phi)
    print(f"x={x_val:+.1f} → y={y:.4f}  [{pattern}]")`,
        solution: `import torch

def relu(z):
    """ReLU : max(0, z)"""
    return torch.clamp(z, min=0)

theta = torch.tensor([[-0.2, 0.4],
                       [-0.9, 0.9],
                       [ 1.1, -0.7]])

phi = torch.tensor([-0.23, -1.3, 1.3, 0.66])

def shallow_forward(x, theta, phi):
    z1 = theta[0, 0] + theta[0, 1] * x
    z2 = theta[1, 0] + theta[1, 1] * x
    z3 = theta[2, 0] + theta[2, 1] * x
    
    h1 = relu(z1)
    h2 = relu(z2)
    h3 = relu(z3)
    
    y = phi[0] + phi[1] * h1 + phi[2] * h2 + phi[3] * h3
    
    pattern = (f"h1={'ON' if h1 > 0 else 'off'}, "
               f"h2={'ON' if h2 > 0 else 'off'}, "
               f"h3={'ON' if h3 > 0 else 'off'}")
    
    return y.item(), pattern

for x_val in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.5]:
    x = torch.tensor(x_val)
    y, pattern = shallow_forward(x, theta, phi)
    print(f"x={x_val:+.1f} → y={y:.4f}  [{pattern}]")`,
        hints: [
          'relu(z) = torch.clamp(z, min=0)',
          'z1 = theta[0, 0] + theta[0, 1] * x  (pré-activation)',
          'y = phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3',
          'Un neurone est ON si sa pré-activation z > 0',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 3 — THÉORIQUE : Homogénéité du ReLU
      // ═══════════════════════════════════════
      {
        id: 'shallow-th2',
        title: '🧠 Théorie — Propriété du ReLU (Prob. 3.5)',
        instructions: 'Problème 3.5 : prouvez numériquement que ReLU(α·z) = α·ReLU(z) pour tout α ≥ 0 (propriété d\'homogénéité non-négative). Puis montrez que cette propriété NE tient PAS pour α < 0. Testez avec différentes valeurs de z et α.',
        starterCode: `import torch

def relu(z):
    return torch.clamp(z, min=0)

# Testez la propriété : ReLU(α·z) == α·ReLU(z) pour α ≥ 0
z_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])

print("═══ α ≥ 0 : propriété VRAIE ═══")
for alpha in [0.0, 0.5, 1.0, 2.0, 10.0]:
    lhs = relu(alpha * z_values)   # ReLU(α·z)
    rhs = alpha * relu(z_values)   # α·ReLU(z)
    equal = torch.allclose(lhs, rhs)
    print(f"  α={alpha:4.1f} : ReLU(αz) = {lhs.tolist()}")
    print(f"          α·ReLU(z) = {rhs.tolist()} → {'✓ ÉGAL' if equal else '✗ DIFFÉRENT'}")

print("\\n═══ α < 0 : propriété FAUSSE ═══")
alpha = -1.0
lhs = relu(alpha * z_values)
rhs = alpha * relu(z_values)
print(f"  α={alpha:4.1f} : ReLU(αz) = {lhs.tolist()}")
print(f"          α·ReLU(z) = {rhs.tolist()}")
print(f"  → {'✓ ÉGAL' if torch.allclose(lhs, rhs) else '✗ DIFFÉRENT'}")

# Question bonus : pourquoi c'est important ?
print("\\n💡 Conséquence (Prob 3.6) :")
print("   Si on multiplie θ₁₀,θ₁₁ par α>0 et divise ϕ₁ par α,")
print("   le réseau donne EXACTEMENT la même sortie.")
print("   → Il y a une infinité de combinaisons de paramètres équivalentes !")`,
        solution: `import torch

def relu(z):
    return torch.clamp(z, min=0)

z_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])

print("═══ α ≥ 0 : propriété VRAIE ═══")
for alpha in [0.0, 0.5, 1.0, 2.0, 10.0]:
    lhs = relu(alpha * z_values)
    rhs = alpha * relu(z_values)
    equal = torch.allclose(lhs, rhs)
    print(f"  α={alpha:4.1f} : ReLU(αz) = {lhs.tolist()}")
    print(f"          α·ReLU(z) = {rhs.tolist()} → {'✓ ÉGAL' if equal else '✗ DIFFÉRENT'}")

print("\\n═══ α < 0 : propriété FAUSSE ═══")
alpha = -1.0
lhs = relu(alpha * z_values)
rhs = alpha * relu(z_values)
print(f"  α={alpha:4.1f} : ReLU(αz) = {lhs.tolist()}")
print(f"          α·ReLU(z) = {rhs.tolist()}")
print(f"  → {'✓ ÉGAL' if torch.allclose(lhs, rhs) else '✗ DIFFÉRENT'}")

print("\\n💡 Conséquence (Prob 3.6) :")
print("   Si on multiplie θ₁₀,θ₁₁ par α>0 et divise ϕ₁ par α,")
print("   le réseau donne EXACTEMENT la même sortie.")
print("   → Il y a une infinité de combinaisons de paramètres équivalentes !")`,
        hints: [
          'Pour α ≥ 0 et z ≥ 0 : ReLU(α·z) = α·z = α·ReLU(z) ✓',
          'Pour α ≥ 0 et z < 0 : ReLU(α·z) = 0 = α·0 = α·ReLU(z) ✓',
          'Pour α < 0 et z > 0 : ReLU(α·z) = 0 ≠ α·z = α·ReLU(z) ✗',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 4 — PRATIQUE : nn.Sequential
      // ═══════════════════════════════════════
      {
        id: 'shallow-pr2',
        title: '💻 Pratique — Réseau PyTorch nn.Sequential',
        instructions: 'Construisez un réseau superficiel avec `nn.Sequential` : 1 entrée, D=20 unités cachées, 1 sortie. Comptez les paramètres, puis passez un batch de 50 entrées à travers le réseau.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

# Construisez le réseau superficiel
D = 20  # unités cachées
model = nn.Sequential(
    ___,  # couche 1 : entrée → cachée
    ___,  # activation ReLU
    ___,  # couche 2 : cachée → sortie
)

# Comptez les paramètres
n_params = ___
print(f"Architecture: {model}")
print(f"Paramètres: {n_params}")

# Vérification théorique
D_i, D_o = 1, 1
n_theorique = D * (D_i + 1) + D_o * (D + 1)
print(f"Formule: {D}×({D_i}+1) + {D_o}×({D}+1) = {n_theorique}")

# Passez un batch de 50 entrées
x = torch.linspace(-3, 3, 50).unsqueeze(1)  # (50, 1)
y = model(x)  # forward pass

print(f"\\nInput shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Premières sorties: {y[:5].squeeze().tolist()}")

# Inspectez les poids de la couche cachée
W1 = model[0].weight  # matrice de poids
b1 = model[0].bias    # vecteur de biais
print(f"\\nPoids couche cachée: {W1.shape} → {W1.numel()} poids")
print(f"Biais couche cachée: {b1.shape} → {b1.numel()} biais")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

D = 20
model = nn.Sequential(
    nn.Linear(1, D),   # couche 1 : entrée → cachée
    nn.ReLU(),          # activation ReLU
    nn.Linear(D, 1),   # couche 2 : cachée → sortie
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {model}")
print(f"Paramètres: {n_params}")

D_i, D_o = 1, 1
n_theorique = D * (D_i + 1) + D_o * (D + 1)
print(f"Formule: {D}×({D_i}+1) + {D_o}×({D}+1) = {n_theorique}")

x = torch.linspace(-3, 3, 50).unsqueeze(1)
y = model(x)

print(f"\\nInput shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Premières sorties: {y[:5].squeeze().tolist()}")

W1 = model[0].weight
b1 = model[0].bias
print(f"\\nPoids couche cachée: {W1.shape} → {W1.numel()} poids")
print(f"Biais couche cachée: {b1.shape} → {b1.numel()} biais")`,
        hints: [
          'nn.Linear(1, D) pour la couche d\'entrée vers cachée',
          'nn.ReLU() comme activation',
          'nn.Linear(D, 1) pour la couche cachée vers sortie',
          'sum(p.numel() for p in model.parameters()) pour le total',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 5 — THÉORIQUE : Compter les régions
      // ═══════════════════════════════════════
      {
        id: 'shallow-th3',
        title: '🧠 Théorie — Compter les régions linéaires (Prob. 3.18)',
        instructions: 'Implémentez la formule de Zaslavsky pour calculer le nombre maximum de régions linéaires. Vérifiez que D=3 en 2D donne 7 régions (comme Fig. 3.8j). Explorez comment le nombre de régions explose en haute dimension.',
        starterCode: `import math

def binomial(n, k):
    """Coefficient binomial C(n, k)"""
    if k > n or k < 0:
        return 0
    return math.comb(n, k)

def max_regions(D, D_i):
    """
    Nombre max de régions d'un shallow network
    D : nombre d'unités cachées
    D_i : dimension de l'entrée
    Formule de Zaslavsky (1975)
    """
    total = ___  # Σ C(D, j) pour j = 0 à min(D, D_i)
    return total

# Vérifications du livre
print("═══ Vérifications ═══")
print(f"D=3, D_i=1 : {max_regions(3, 1)} régions (attendu: 4)")
print(f"D=3, D_i=2 : {max_regions(3, 2)} régions (attendu: 7)")
print(f"D=5, D_i=2 : {max_regions(5, 2)} régions (attendu: 16)")

# Exploration
print("\\n═══ Explosion combinatoire ═══")
for D_i in [1, 2, 5, 10, 50, 100]:
    D = max(D_i, 10)
    r = max_regions(D, D_i)
    print(f"D_i={D_i:3d}, D={D:3d} → {r:.2e} régions max")

# Cas massif du livre
D, D_i = 500, 100
r = max_regions(D, D_i)
print(f"\\nD=500, D_i=100 → ~10^{math.log10(r):.0f} régions !")`,
        solution: `import math

def binomial(n, k):
    if k > n or k < 0:
        return 0
    return math.comb(n, k)

def max_regions(D, D_i):
    total = sum(binomial(D, j) for j in range(min(D, D_i) + 1))
    return total

print("═══ Vérifications ═══")
print(f"D=3, D_i=1 : {max_regions(3, 1)} régions (attendu: 4)")
print(f"D=3, D_i=2 : {max_regions(3, 2)} régions (attendu: 7)")
print(f"D=5, D_i=2 : {max_regions(5, 2)} régions (attendu: 16)")

print("\\n═══ Explosion combinatoire ═══")
for D_i in [1, 2, 5, 10, 50, 100]:
    D = max(D_i, 10)
    r = max_regions(D, D_i)
    print(f"D_i={D_i:3d}, D={D:3d} → {r:.2e} régions max")

D, D_i = 500, 100
r = max_regions(D, D_i)
print(f"\\nD=500, D_i=100 → ~10^{math.log10(r):.0f} régions !")`,
        hints: [
          'La formule est : Σ C(D, j) pour j de 0 à min(D, D_i)',
          'sum(binomial(D, j) for j in range(min(D, D_i) + 1))',
          'Pour D_i=1 : C(D,0) + C(D,1) = 1 + D = D+1 ✓',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 6 — PRATIQUE : Entrainer un shallow network
      // ═══════════════════════════════════════
      {
        id: 'shallow-pr3',
        title: '💻 Pratique — Entraîner un shallow network',
        instructions: 'Entraînez un réseau superficiel pour approximer la fonction sin(x) sur [-π, π]. Observez comment le nombre d\'unités cachées D affecte la qualité de l\'approximation.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

# Données : y = sin(x) sur [-π, π]
x_train = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y_train = torch.sin(x_train)

def train_shallow(D, n_epochs=2000, lr=0.01):
    """Entraîne un réseau superficiel avec D unités cachées"""
    model = nn.Sequential(
        ___,   # entrée → D unités
        ___,   # ReLU
        ___,   # D unités → sortie
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(n_epochs):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss_fn(model(x_train), y_train).item()
    return model, final_loss

# Comparer différentes capacités
print("D (units)  │  Params  │  MSE Loss")
print("───────────┼──────────┼──────────")
for D in [3, 5, 10, 20, 50]:
    model, loss = train_shallow(D)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  D={D:3d}     │  {n_params:5d}   │  {loss:.6f}")

print("\\n→ Plus de hidden units = meilleure approximation de sin(x)")
print("  Cela illustre le théorème d'approximation universelle !")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

x_train = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y_train = torch.sin(x_train)

def train_shallow(D, n_epochs=2000, lr=0.01):
    model = nn.Sequential(
        nn.Linear(1, D),
        nn.ReLU(),
        nn.Linear(D, 1),
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(n_epochs):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss_fn(model(x_train), y_train).item()
    return model, final_loss

print("D (units)  │  Params  │  MSE Loss")
print("───────────┼──────────┼──────────")
for D in [3, 5, 10, 20, 50]:
    model, loss = train_shallow(D)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  D={D:3d}     │  {n_params:5d}   │  {loss:.6f}")

print("\\n→ Plus de hidden units = meilleure approximation de sin(x)")
print("  Cela illustre le théorème d'approximation universelle !")`,
        hints: [
          'nn.Linear(1, D) pour la couche d\'entrée',
          'nn.ReLU() pour l\'activation',
          'nn.Linear(D, 1) pour la couche de sortie',
          'Adam converge plus vite que SGD pour ce problème',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 7 — THÉORIQUE : Pentes des régions
      // ═══════════════════════════════════════
      {
        id: 'shallow-th4',
        title: '🧠 Théorie — Pentes des régions lin. (Prob. 3.3)',
        instructions: 'Problème 3.3 : calculez les positions des joints et les pentes de chaque région linéaire. Montrez que la somme d\'une pente intermédiaire est la somme des pentes des unités actives dans cette région.',
        starterCode: `import torch

# Paramètres de la Figure 3.2a du livre
theta = torch.tensor([[-0.2, 0.4],   # θ₁₀, θ₁₁
                       [-0.9, 0.9],   # θ₂₀, θ₂₁
                       [ 1.1, -0.7]]) # θ₃₀, θ₃₁

phi = torch.tensor([-0.23, -1.3, 1.3, 0.66])

# ── Positions des joints ──
# Un joint est là où θ_{d0} + θ_{d1}·x = 0
# Donc x_joint = -θ_{d0} / θ_{d1}
joint1 = ___  # = -theta[0,0] / theta[0,1]
joint2 = ___
joint3 = ___

print("═══ Positions des joints ═══")
print(f"Joint 1 (h₁ s'active) : x = {joint1:.4f}")
print(f"Joint 2 (h₂ s'active) : x = {joint2:.4f}")
print(f"Joint 3 (h₃ s'active) : x = {joint3:.4f}")

# Triez les joints pour identifier les régions
joints = sorted([(joint1.item(), 1), (joint2.item(), 2), (joint3.item(), 3)])
print(f"\\nJoints triés: {[(f'x={j:.2f}', f'h{i}') for j, i in joints]}")

# ── Pentes des régions ──
# La pente d'une région = Σ (ϕ_d · θ_{d1}) pour chaque unité d ACTIVE
print("\\n═══ Pentes des régions ═══")
print(f"ϕ₁·θ₁₁ = {phi[1]*theta[0,1]:.4f}")
print(f"ϕ₂·θ₂₁ = {phi[2]*theta[1,1]:.4f}")
print(f"ϕ₃·θ₃₁ = {phi[3]*theta[2,1]:.4f}")

# Identifiez quelles unités sont actives dans chaque région
# et calculez la pente correspondante`,
        solution: `import torch

theta = torch.tensor([[-0.2, 0.4],
                       [-0.9, 0.9],
                       [ 1.1, -0.7]])

phi = torch.tensor([-0.23, -1.3, 1.3, 0.66])

joint1 = -theta[0, 0] / theta[0, 1]
joint2 = -theta[1, 0] / theta[1, 1]
joint3 = -theta[2, 0] / theta[2, 1]

print("═══ Positions des joints ═══")
print(f"Joint 1 (h₁ s'active) : x = {joint1:.4f}")
print(f"Joint 2 (h₂ s'active) : x = {joint2:.4f}")
print(f"Joint 3 (h₃ s'active) : x = {joint3:.4f}")

joints = sorted([(joint1.item(), 1), (joint2.item(), 2), (joint3.item(), 3)])
print(f"\\nJoints triés: {[(f'x={j:.2f}', f'h{i}') for j, i in joints]}")

print("\\n═══ Pentes des régions ═══")
print(f"ϕ₁·θ₁₁ = {phi[1]*theta[0,1]:.4f}")
print(f"ϕ₂·θ₂₁ = {phi[2]*theta[1,1]:.4f}")
print(f"ϕ₃·θ₃₁ = {phi[3]*theta[2,1]:.4f}")`,
        hints: [
          'Le joint du neurone d est à x = -θ_{d0} / θ_{d1}',
          'La pente dans une région = somme de ϕ_d · θ_{d1} pour les neurones actifs',
          'joint1 = -theta[0, 0] / theta[0, 1]',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 8 — PRATIQUE : Comparer les activations
      // ═══════════════════════════════════════
      {
        id: 'shallow-pr4',
        title: '💻 Pratique — Comparer ReLU, Sigmoid, Tanh, GELU',
        instructions: 'Créez 4 réseaux superficiels identiques (D=10) mais avec des activations différentes : ReLU, Sigmoid, Tanh, GELU. Entraînez-les sur y=sin(x) et comparez les pertes finales.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(0)

# Données
x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y = torch.sin(x)

D = 10  # unités cachées

# Dictionnaire des activations à tester
activations = {
    'ReLU':    ___,
    'Sigmoid': ___,
    'Tanh':    ___,
    'GELU':    ___,
}

results = {}

for name, act_fn in activations.items():
    torch.manual_seed(0)  # même init pour comparaison juste
    
    model = nn.Sequential(
        nn.Linear(1, D),
        act_fn,
        nn.Linear(D, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    for epoch in range(1000):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    results[name] = final_loss

# Affichage
print("═══ Comparaison des activations (D=10, 1000 epochs) ═══")
print(f"{'Activation':<12} │ {'MSE Loss':>10}")
print(f"{'─'*12}─┼─{'─'*10}")
for name, loss in sorted(results.items(), key=lambda x: x[1]):
    bar = '█' * int(min(50, 50 * (1 - loss/max(results.values()))))
    print(f"{name:<12} │ {loss:10.6f}  {bar}")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(0)

x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y = torch.sin(x)

D = 10

activations = {
    'ReLU':    nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh':    nn.Tanh(),
    'GELU':    nn.GELU(),
}

results = {}

for name, act_fn in activations.items():
    torch.manual_seed(0)
    
    model = nn.Sequential(
        nn.Linear(1, D),
        act_fn,
        nn.Linear(D, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    for epoch in range(1000):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    results[name] = final_loss

print("═══ Comparaison des activations (D=10, 1000 epochs) ═══")
print(f"{'Activation':<12} │ {'MSE Loss':>10}")
print(f"{'─'*12}─┼─{'─'*10}")
for name, loss in sorted(results.items(), key=lambda x: x[1]):
    bar = '█' * int(min(50, 50 * (1 - loss/max(results.values()))))
    print(f"{name:<12} │ {loss:10.6f}  {bar}")`,
        hints: [
          'nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU()',
          'Utilisez torch.manual_seed(0) avant chaque modèle pour la reproductibilité',
          'GELU et Tanh tendent à mieux approximer les fonctions lisses que ReLU',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 9 — PRATIQUE : nn.Module personnalisé
      // ═══════════════════════════════════════
      {
        id: 'shallow-pr5',
        title: '💻 Pratique — nn.Module personnalisé',
        instructions: 'Implémentez un réseau superficiel en tant que classe `nn.Module` personnalisée (pas `nn.Sequential`). Ajoutez une méthode `count_params()` et une méthode `get_activation_pattern(x)` qui retourne quelles unités sont actives.',
        starterCode: `import torch
import torch.nn as nn

class ShallowNetwork(nn.Module):
    def __init__(self, D_i, D, D_o):
        super().__init__()
        self.hidden = ___     # nn.Linear(D_i, D)
        self.output = ___     # nn.Linear(D, D_o)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass : x → ReLU(Wx+b) → sortie"""
        h = ___   # couche cachée + activation
        y = ___   # couche de sortie
        return y
    
    def count_params(self):
        """Retourne le nombre total de paramètres"""
        return ___
    
    def get_activation_pattern(self, x):
        """Retourne un tenseur binaire (1=actif, 0=inactif)"""
        pre_act = self.hidden(x)       # pré-activations
        pattern = (pre_act > 0).int()  # 1 si actif, 0 sinon
        return pattern

# Créez et testez le réseau
model = ShallowNetwork(D_i=2, D=5, D_o=1)

print(f"Architecture: {model}")
print(f"Paramètres: {model.count_params()}")

# Test
x = torch.tensor([[1.0, -0.5]])
y = model(x)
pattern = model.get_activation_pattern(x)

print(f"\\nEntrée:  {x.tolist()}")
print(f"Sortie:  {y.item():.4f}")
print(f"Pattern: {pattern.tolist()[0]} (1=ON, 0=off)")
print(f"Unités actives: {pattern.sum().item()}/{model.hidden.out_features}")`,
        solution: `import torch
import torch.nn as nn

class ShallowNetwork(nn.Module):
    def __init__(self, D_i, D, D_o):
        super().__init__()
        self.hidden = nn.Linear(D_i, D)
        self.output = nn.Linear(D, D_o)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.hidden(x))
        y = self.output(h)
        return y
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_activation_pattern(self, x):
        pre_act = self.hidden(x)
        pattern = (pre_act > 0).int()
        return pattern

model = ShallowNetwork(D_i=2, D=5, D_o=1)

print(f"Architecture: {model}")
print(f"Paramètres: {model.count_params()}")

x = torch.tensor([[1.0, -0.5]])
y = model(x)
pattern = model.get_activation_pattern(x)

print(f"\\nEntrée:  {x.tolist()}")
print(f"Sortie:  {y.item():.4f}")
print(f"Pattern: {pattern.tolist()[0]} (1=ON, 0=off)")
print(f"Unités actives: {pattern.sum().item()}/{model.hidden.out_features}")`,
        hints: [
          'self.hidden = nn.Linear(D_i, D)',
          'h = self.relu(self.hidden(x)) — appliquer ReLU aux pré-activations',
          'count_params: sum(p.numel() for p in self.parameters())',
        ],
        completed: false,
      },

      // ═══════════════════════════════════════
      // EXERCICE 10 — THÉORIQUE : Unicité de la solution
      // ═══════════════════════════════════════
      {
        id: 'shallow-th5',
        title: '🧠 Théorie — Unicité de la solution (Prob. 3.7)',
        instructions: 'Problème 3.7 : la perte des moindres carrés a-t-elle un minimum unique ? Montrez qu\'il existe une infinité de combinaisons de paramètres qui donnent exactement la même fonction (même perte). Démontrez-le en construisant 2 réseaux avec des paramètres différents mais la même sortie.',
        starterCode: `import torch

def relu(z):
    return torch.clamp(z, min=0)

def shallow_net(x, theta, phi):
    h1 = relu(theta[0,0] + theta[0,1] * x)
    h2 = relu(theta[1,0] + theta[1,1] * x)
    h3 = relu(theta[2,0] + theta[2,1] * x)
    return phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3

# Réseau A — paramètres originaux
theta_A = torch.tensor([[-0.2, 0.4], [-0.9, 0.9], [1.1, -0.7]])
phi_A   = torch.tensor([-0.23, -1.3, 1.3, 0.66])

# Réseau B — MÊMES résultats mais paramètres DIFFÉRENTS
# Astuce 1 : multiplier θ₁ par α et diviser ϕ₁ par α (Prob. 3.6)
alpha = 2.0
theta_B = theta_A.clone()
theta_B[0] = theta_A[0] * alpha  # multiplie θ₁₀, θ₁₁ par α
phi_B = phi_A.clone()
phi_B[1] = phi_A[1] / alpha      # divise ϕ₁ par α

# Astuce 2 : permuter les unités cachées (h₁ ↔ h₂)
theta_C = torch.tensor([[-0.9, 0.9], [-0.2, 0.4], [1.1, -0.7]])  # permutation
phi_C   = torch.tensor([-0.23, 1.3, -1.3, 0.66])  # ϕ₁ ↔ ϕ₂ échangés

# Vérification
x_test = torch.linspace(-2, 3, 10)
print("   x    │   Net A   │   Net B   │   Net C   │  B=A?  C=A?")
print("────────┼───────────┼───────────┼───────────┼────────────")
for x in x_test:
    yA = shallow_net(x, theta_A, phi_A).item()
    yB = shallow_net(x, theta_B, phi_B).item()
    yC = shallow_net(x, theta_C, phi_C).item()
    eq_B = "✓" if abs(yA - yB) < 1e-5 else "✗"
    eq_C = "✓" if abs(yA - yC) < 1e-5 else "✗"
    print(f"  {x:+.2f}  │  {yA:+.4f}  │  {yB:+.4f}  │  {yC:+.4f}  │  {eq_B}     {eq_C}")

print("\\n💡 Conclusion : le minimum de la perte N'EST PAS unique !")
print("   → Il existe des symétries : scaling (×α) et permutation.")
print("   → Le paysage de perte a une infinité de minima globaux équivalents.")`,
        solution: `import torch

def relu(z):
    return torch.clamp(z, min=0)

def shallow_net(x, theta, phi):
    h1 = relu(theta[0,0] + theta[0,1] * x)
    h2 = relu(theta[1,0] + theta[1,1] * x)
    h3 = relu(theta[2,0] + theta[2,1] * x)
    return phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3

theta_A = torch.tensor([[-0.2, 0.4], [-0.9, 0.9], [1.1, -0.7]])
phi_A   = torch.tensor([-0.23, -1.3, 1.3, 0.66])

alpha = 2.0
theta_B = theta_A.clone()
theta_B[0] = theta_A[0] * alpha
phi_B = phi_A.clone()
phi_B[1] = phi_A[1] / alpha

theta_C = torch.tensor([[-0.9, 0.9], [-0.2, 0.4], [1.1, -0.7]])
phi_C   = torch.tensor([-0.23, 1.3, -1.3, 0.66])

x_test = torch.linspace(-2, 3, 10)
print("   x    │   Net A   │   Net B   │   Net C   │  B=A?  C=A?")
print("────────┼───────────┼───────────┼───────────┼────────────")
for x in x_test:
    yA = shallow_net(x, theta_A, phi_A).item()
    yB = shallow_net(x, theta_B, phi_B).item()
    yC = shallow_net(x, theta_C, phi_C).item()
    eq_B = "✓" if abs(yA - yB) < 1e-5 else "✗"
    eq_C = "✓" if abs(yA - yC) < 1e-5 else "✗"
    print(f"  {x:+.2f}  │  {yA:+.4f}  │  {yB:+.4f}  │  {yC:+.4f}  │  {eq_B}     {eq_C}")

print("\\n💡 Conclusion : le minimum de la perte N'EST PAS unique !")
print("   → Il existe des symétries : scaling (×α) et permutation.")
print("   → Le paysage de perte a une infinité de minima globaux équivalents.")`,
        hints: [
          'Par homogénéité du ReLU : ReLU(α·z) = α·ReLU(z) pour α > 0',
          'Donc (α·θ) passé par ReLU puis (ϕ/α) = même résultat',
          'Permuter les unités cachées revient à réarranger les indices',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Réseaux Superficiels — Ch. 3 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Implémentation manuelle (Éq. 3.1) ──
def shallow_network_manual(x, theta, phi):
    """
    Réseau superficiel : y = ϕ₀ + Σ ϕd · ReLU(θd₀ + θd₁·x)
    
    x:     entrée scalaire (tensor)
    theta: paramètres couche cachée — shape (D, 2)
           theta[d, 0] = biais,  theta[d, 1] = pente
    phi:   paramètres de sortie — shape (D+1,)
           phi[0] = offset, phi[1:] = poids de recombinaison
    """
    D = theta.shape[0]
    
    # Pré-activations : z_d = θ_{d0} + θ_{d1}·x
    z = theta[:, 0] + theta[:, 1] * x
    
    # Activations : h_d = ReLU(z_d)
    h = torch.relu(z)
    
    # Sortie : y = ϕ₀ + Σ ϕ_d · h_d
    y = phi[0] + torch.sum(phi[1:] * h)
    
    return y, z, h

# Paramètres du livre (Figure 3.2a)
theta = torch.tensor([[-0.2, 0.4],
                       [-0.9, 0.9],
                       [ 1.1, -0.7]])
phi = torch.tensor([-0.23, -1.3, 1.3, 0.66])

print("── Forward pass manuel ──")
for x_val in [-1.5, 0.0, 0.5, 1.0, 2.0]:
    x = torch.tensor(x_val)
    y, z, h = shallow_network_manual(x, theta, phi)
    active = ["ON" if hi > 0 else "off" for hi in h]
    print(f"x={x_val:+.1f} → y={y:.4f}  pattern=[{', '.join(active)}]")

# ── 2. Même réseau avec PyTorch nn.Module ──
print("\\n── Réseau PyTorch nn.Sequential ──")

# Initialiser avec les MÊMES paramètres
model = nn.Sequential(
    nn.Linear(1, 3),  # couche cachée
    nn.ReLU(),
    nn.Linear(3, 1),  # couche de sortie
)

# Copier nos paramètres dans le modèle PyTorch
with torch.no_grad():
    model[0].weight.copy_(theta[:, 1:])  # pentes
    model[0].bias.copy_(theta[:, 0])     # biais
    model[2].weight.copy_(phi[1:].unsqueeze(0))
    model[2].bias.copy_(phi[:1])

for x_val in [-1.5, 0.0, 0.5, 1.0, 2.0]:
    x = torch.tensor([[x_val]])
    y = model(x)
    print(f"x={x_val:+.1f} → y={y.item():.4f}")

# ── 3. Comptage de paramètres ──
print(f"\\n── Paramètres ──")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} = {param.numel()} params")
print(f"  Total: {sum(p.numel() for p in model.parameters())}")
print(f"  Formule: D(D_i+1) + D_o(D+1) = 3(1+1) + 1(3+1) = 10 ✓")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 4 — RÉSEAUX PROFONDS
  // ═══════════════════════════════════════
  {
    id: 'deep-networks',
    title: 'Réseaux de Neurones Profonds',
    shortTitle: 'Deep Nets',
    description: 'Composition de réseaux, profondeur vs largeur, et notation matricielle.',
    status: 'locked',
    progress: 0,
    dependencies: ['shallow-networks'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Un **réseau profond** est obtenu en **composant** plusieurs réseaux superficiels : la sortie du premier devient l'entrée du second, et ainsi de suite. Cette composition crée des fonctions beaucoup plus complexes.\n\nAvec ReLU, un réseau profond de K couches de D unités cachées chacune peut créer jusqu'à **(D+1)^K** régions linéaires, contre D+1 pour un réseau superficiel.`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}_k = a[\\boldsymbol{\\beta}_k + \\boldsymbol{\\Omega}_k \\mathbf{h}_{k-1}]',
        label: 'Couche k du réseau profond',
        highlightVar: 'hidden',
      },
      {
        type: 'text',
        content: `En notation matricielle, chaque couche applique une transformation affine (multiplication par la matrice de poids **Ωk** + biais **βk**) suivie d'une activation. Le réseau complet est :\n\n- Forward pass : on calcule séquentiellement h₁, h₂, ..., hK\n- La sortie finale f₃ est le résultat de la dernière couche`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{f}_0 &= \\boldsymbol{\\beta}_0 + \\boldsymbol{\\Omega}_0 \\mathbf{x} \\\\ \\mathbf{h}_k &= a[\\mathbf{f}_{k-1}] \\\\ \\mathbf{f}_k &= \\boldsymbol{\\beta}_k + \\boldsymbol{\\Omega}_k \\mathbf{h}_k \\end{aligned}',
        label: 'Forward pass complet',
        highlightVar: 'hidden',
      },
      {
        type: 'callout',
        content: '⚡ **Profondeur vs Largeur** : un réseau profond avec le même nombre total de paramètres qu\'un réseau superficiel large peut représenter des fonctions exponentiellement plus complexes. C\'est pourquoi le "deep" learning est si puissant.',
      },
    ],
    exercises: [
      {
        id: 'deep-ex1',
        title: 'Construire un réseau à 3 couches',
        instructions: 'Créez un réseau nn.Module avec 3 couches cachées (784→256→128→64→10). Comptez les paramètres.',
        starterCode: `import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ___
        self.layer2 = ___
        self.layer3 = ___
        self.output = ___
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)

model = DeepNet()
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal paramètres: {total:,}")`,
        solution: `import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)

model = DeepNet()
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal paramètres: {total:,}")`,
        hints: [
          'nn.Linear(in_features, out_features) crée une couche dense',
          'Couche 1: 784→256, Couche 2: 256→128, Couche 3: 128→64, Sortie: 64→10',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Réseaux Profonds vs Superficiels ══

# Réseau SUPERFICIEL (1 couche cachée large)
shallow = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

# Réseau PROFOND (3 couches cachées étroites)
deep = nn.Sequential(
    nn.Linear(1, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

shallow_params = sum(p.numel() for p in shallow.parameters())
deep_params = sum(p.numel() for p in deep.parameters())

print(f"Shallow: {shallow_params} paramètres")
print(f"Deep:    {deep_params} paramètres")

# Les deux ont ~300 paramètres mais le deep peut
# représenter des fonctions beaucoup plus complexes !

# Forward pass
x = torch.randn(5, 1)
print(f"\\nShallow output: {shallow(x).squeeze().tolist()}")
print(f"Deep output:    {deep(x).squeeze().tolist()}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 5 — FONCTIONS DE PERTE
  // ═══════════════════════════════════════
  {
    id: 'loss-functions',
    title: 'Fonctions de Perte (Loss Functions)',
    shortTitle: 'Loss',
    description: 'Maximum de vraisemblance, MSE, Cross-Entropy — mesurer l\'erreur du modèle.',
    status: 'locked',
    progress: 0,
    dependencies: ['deep-networks'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `La **fonction de perte** mesure à quel point les prédictions sont éloignées de la réalité. En Deep Learning, on construit les fonctions de perte via le **maximum de vraisemblance** : le modèle prédit une distribution de probabilité Pr(y|x), et on cherche les paramètres qui maximisent la probabilité des données observées.`,
      },
      {
        type: 'equation',
        content: '\\hat{\\boldsymbol{\\phi}} = \\underset{\\boldsymbol{\\phi}}{\\text{argmax}} \\prod_{i=1}^{I} Pr(y_i | x_i)',
        label: 'Maximum de vraisemblance',
      },
      {
        type: 'text',
        content: `En prenant le logarithme négatif (pour transformer le produit en somme et la maximisation en minimisation), on obtient la **perte de log-vraisemblance négative**. Pour la régression avec bruit gaussien, cela donne le **MSE** (Mean Squared Error) :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{MSE} = \\frac{1}{I}\\sum_{i=1}^{I}(y_i - f[x_i, \\boldsymbol{\\phi}])^2',
        label: 'Mean Squared Error (Régression)',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `Pour la **classification binaire**, le modèle prédit une probabilité via sigmoid, et la perte est la **cross-entropie binaire**. Pour la classification **multi-classe** avec K classes, on utilise softmax + cross-entropie :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{CE} = -\\sum_{i=1}^{I} \\sum_{k=1}^{K} y_{ik} \\log(\\hat{y}_{ik})',
        label: 'Cross-Entropy Loss (Classification)',
        highlightVar: 'loss',
      },
      {
        type: 'callout',
        content: '⚡ La **cross-entropie** mesure la "distance" entre la distribution prédite et la distribution réelle. Elle est toujours ≥ 0 et vaut 0 seulement quand la prédiction est parfaite.',
      },
    ],
    exercises: [
      {
        id: 'loss-ex1',
        title: 'Comparer MSE et Cross-Entropy',
        instructions: 'Calculez la perte MSE pour un problème de régression et la perte Cross-Entropy pour un problème de classification.',
        starterCode: `import torch
import torch.nn as nn

# ── Régression : MSE ──
predictions = torch.tensor([2.5, 3.2, 4.1])
targets = torch.tensor([3.0, 3.0, 4.0])

mse = nn.MSELoss()
loss_mse = ___

# ── Classification : Cross-Entropy ──
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])

ce = nn.CrossEntropyLoss()
loss_ce = ___

print(f"MSE Loss: {loss_mse.item():.4f}")
print(f"CE Loss:  {loss_ce.item():.4f}")`,
        solution: `import torch
import torch.nn as nn

# ── Régression : MSE ──
predictions = torch.tensor([2.5, 3.2, 4.1])
targets = torch.tensor([3.0, 3.0, 4.0])

mse = nn.MSELoss()
loss_mse = mse(predictions, targets)

# ── Classification : Cross-Entropy ──
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])

ce = nn.CrossEntropyLoss()
loss_ce = ce(logits, labels)

print(f"MSE Loss: {loss_mse.item():.4f}")
print(f"CE Loss:  {loss_ce.item():.4f}")`,
        hints: [
          'loss_mse = mse(predictions, targets)',
          'CrossEntropyLoss prend des logits (avant softmax) et des labels entiers',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Fonctions de Perte — Depuis le Maximum de Vraisemblance ══

# ── 1. MSE pour la régression ──
pred = torch.tensor([2.5, 3.2, 4.1, 1.8])
target = torch.tensor([3.0, 3.0, 4.0, 2.0])

mse = nn.MSELoss()
print(f"MSE Loss: {mse(pred, target).item():.4f}")

# Calcul manuel
manual_mse = torch.mean((pred - target) ** 2)
print(f"MSE manuelle: {manual_mse.item():.4f}")

# ── 2. Binary Cross-Entropy ──
# Pour classification binaire (sortie sigmoid)
pred_prob = torch.tensor([0.9, 0.2, 0.8])
target_bin = torch.tensor([1.0, 0.0, 1.0])

bce = nn.BCELoss()
print(f"\\nBCE Loss: {bce(pred_prob, target_bin).item():.4f}")

# ── 3. Cross-Entropy pour multi-classe ──
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [0.1, 0.3, 3.0]])
labels = torch.tensor([0, 1, 2])  # classes correctes

ce = nn.CrossEntropyLoss()
print(f"CE Loss: {ce(logits, labels).item():.4f}")

# Vérifier les probabilités avec softmax
probs = torch.softmax(logits, dim=1)
print(f"\\nProbabilités prédites:\\n{probs}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 6 — DESCENTE DE GRADIENT
  // ═══════════════════════════════════════
  {
    id: 'gradient-descent',
    title: 'Descente de Gradient & Optimisation',
    shortTitle: 'Gradient',
    description: 'Gradient descent, SGD, Momentum et Adam — comment entraîner un réseau.',
    status: 'locked',
    progress: 0,
    dependencies: ['loss-functions'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `La **descente de gradient** est l'algorithme itératif standard pour entraîner les réseaux de neurones. On part de paramètres initiaux aléatoires, puis on répète deux étapes :\n\n1. **Calculer le gradient** ∂L/∂ϕ de la perte par rapport aux paramètres\n2. **Mettre à jour** les paramètres dans la direction opposée au gradient`,
      },
      {
        type: 'equation',
        content: '\\boldsymbol{\\phi} \\leftarrow \\boldsymbol{\\phi} - \\alpha \\cdot \\frac{\\partial \\mathcal{L}}{\\partial \\boldsymbol{\\phi}}',
        label: 'Règle de mise à jour (Gradient Descent)',
        highlightVar: 'grad',
      },
      {
        type: 'text',
        content: `Le **SGD** (Stochastic Gradient Descent) utilise un **mini-batch** aléatoire au lieu de tout le dataset, ce qui est bien plus rapide. Le **Momentum** ajoute une "inertie" qui lisse les mises à jour. **Adam** combine momentum + adaptation du learning rate par paramètre.`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{m}_t &= \\beta_1 \\mathbf{m}_{t-1} + (1-\\beta_1) \\mathbf{g}_t \\\\ \\mathbf{v}_t &= \\beta_2 \\mathbf{v}_{t-1} + (1-\\beta_2) \\mathbf{g}_t^2 \\\\ \\boldsymbol{\\phi}_t &= \\boldsymbol{\\phi}_{t-1} - \\alpha \\frac{\\hat{\\mathbf{m}}_t}{\\sqrt{\\hat{\\mathbf{v}}_t} + \\epsilon} \\end{aligned}',
        label: 'Algorithme Adam',
      },
      {
        type: 'callout',
        content: '💡 Le **learning rate α** est l\'hyperparamètre le plus important. Trop grand → la perte diverge. Trop petit → l\'entraînement est trop lent. Adam (α ≈ 0.001) est le choix par défaut en pratique.',
      },
    ],
    exercises: [
      {
        id: 'gd-ex1',
        title: 'SGD vs Adam',
        instructions: 'Entraînez le même modèle avec SGD et Adam. Comparez la vitesse de convergence.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Données : y = 3x + 2
x = torch.randn(100, 1)
y = 3 * x + 2 + torch.randn(100, 1) * 0.1

# Modèle avec SGD
model_sgd = nn.Linear(1, 1)
opt_sgd = ___

# Modèle avec Adam
model_adam = nn.Linear(1, 1)
opt_adam = ___

loss_fn = nn.MSELoss()

for epoch in range(200):
    # SGD step
    loss_sgd = loss_fn(model_sgd(x), y)
    opt_sgd.zero_grad()
    loss_sgd.backward()
    opt_sgd.step()
    
    # Adam step
    loss_adam = loss_fn(model_adam(x), y)
    opt_adam.zero_grad()
    loss_adam.backward()
    opt_adam.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: SGD={loss_sgd.item():.4f}, Adam={loss_adam.item():.4f}")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x = torch.randn(100, 1)
y = 3 * x + 2 + torch.randn(100, 1) * 0.1

model_sgd = nn.Linear(1, 1)
opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)

model_adam = nn.Linear(1, 1)
opt_adam = optim.Adam(model_adam.parameters(), lr=0.01)

loss_fn = nn.MSELoss()

for epoch in range(200):
    loss_sgd = loss_fn(model_sgd(x), y)
    opt_sgd.zero_grad()
    loss_sgd.backward()
    opt_sgd.step()
    
    loss_adam = loss_fn(model_adam(x), y)
    opt_adam.zero_grad()
    loss_adam.backward()
    opt_adam.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: SGD={loss_sgd.item():.4f}, Adam={loss_adam.item():.4f}")`,
        hints: [
          'optim.SGD(model.parameters(), lr=0.01)',
          'optim.Adam(model.parameters(), lr=0.01)',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn
import torch.optim as optim

# ══ Descente de Gradient — SGD, Momentum, Adam ══

torch.manual_seed(42)

# Données synthétiques : y = 3x + 2 + bruit
x = torch.randn(200, 1)
y = 3 * x + 2 + torch.randn(200, 1) * 0.1

model = nn.Linear(1, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Boucle d'entraînement
for epoch in range(200):
    # Mini-batch (ici on utilise tout le dataset)
    pred = model(x)
    loss = loss_fn(pred, y)
    
    # Gradient → mise à jour
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        w, b = model.weight.item(), model.bias.item()
        print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, y={w:.3f}x + {b:.3f}")

w, b = model.weight.item(), model.bias.item()
print(f"\\n✓ Appris : y = {w:.2f}x + {b:.2f}")
print(f"  Réel  : y = 3.00x + 2.00")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 7 — BACKPROPAGATION
  // ═══════════════════════════════════════
  {
    id: 'backprop',
    title: 'Backpropagation & Autograd',
    shortTitle: 'Backprop',
    description: 'Propagation arrière du gradient — le cœur de l\'apprentissage.',
    status: 'locked',
    progress: 0,
    dependencies: ['gradient-descent'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `La **backpropagation** est l'algorithme qui calcule efficacement les gradients dans un réseau profond. Elle utilise la **règle de la chaîne** pour propager le gradient de la perte depuis la sortie jusqu'aux entrées, couche par couche.\n\nLe processus se déroule en deux passes :\n1. **Forward pass** : calcul de toutes les activations h₁, h₂, ... et de la perte\n2. **Backward pass** : propagation des gradients de la sortie vers l'entrée`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\ell}{\\partial \\mathbf{f}_k} = \\frac{\\partial \\mathbf{h}_{k+1}}{\\partial \\mathbf{f}_k} \\cdot \\frac{\\partial \\mathbf{f}_{k+1}}{\\partial \\mathbf{h}_{k+1}} \\cdot \\frac{\\partial \\ell}{\\partial \\mathbf{f}_{k+1}}',
        label: 'Règle de la chaîne (Backprop)',
        highlightVar: 'grad',
      },
      {
        type: 'text',
        content: `Pour chaque couche k, on calcule les gradients par rapport aux poids **Ωk** et biais **βk** :\n\n- ∂ℓ/∂Ωk = ∂ℓ/∂fk · hkᵀ\n- ∂ℓ/∂βk = ∂ℓ/∂fk\n\nLa dérivée de ReLU est simple : elle vaut 1 si l'entrée > 0, et 0 sinon.`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\, \\text{ReLU}(z)}{\\partial z} = \\begin{cases} 0 & z < 0 \\\\ 1 & z > 0 \\end{cases}',
        label: 'Dérivée du ReLU',
      },
      {
        type: 'callout',
        content: '⚡ PyTorch implémente la backpropagation automatiquement via **Autograd**. Il suffit d\'appeler `loss.backward()` et les gradients sont calculés pour tous les paramètres avec `requires_grad=True`.',
      },
      {
        type: 'text',
        content: `L'**initialisation des paramètres** est cruciale. Si les poids sont trop grands, les gradients explosent. Trop petits, ils s'évanouissent. L'initialisation de **He** (pour ReLU) choisit les poids avec une variance de 2/n, où n est le nombre d'entrées.`,
      },
    ],
    exercises: [
      {
        id: 'bp-ex1',
        title: 'Autograd en action',
        instructions: 'Utilisez PyTorch Autograd pour calculer les gradients d\'une expression. Vérifiez les résultats manuellement.',
        starterCode: `import torch

# Variables avec suivi du gradient
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward : y = w*x + b, loss = (y - 10)²
y = ___
loss = ___

print(f"y = {y.item():.2f}")
print(f"loss = {loss.item():.2f}")

# Backward
loss.backward()

print(f"\\n∂loss/∂w = {w.grad.item():.2f}")
print(f"∂loss/∂x = {x.grad.item():.2f}")
print(f"∂loss/∂b = {b.grad.item():.2f}")

# Vérification manuelle :
# ∂loss/∂y = 2(y-10), ∂y/∂w = x → ∂loss/∂w = 2(y-10)*x
print(f"\\nVérification: 2*(y-10)*x = {2*(y.item()-10)*x.item():.2f}")`,
        solution: `import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b
loss = (y - 10) ** 2

print(f"y = {y.item():.2f}")
print(f"loss = {loss.item():.2f}")

loss.backward()

print(f"\\n∂loss/∂w = {w.grad.item():.2f}")
print(f"∂loss/∂x = {x.grad.item():.2f}")
print(f"∂loss/∂b = {b.grad.item():.2f}")

print(f"\\nVérification: 2*(y-10)*x = {2*(y.item()-10)*x.item():.2f}")`,
        hints: [
          'y = w * x + b',
          'loss = (y - 10) ** 2',
          'loss.backward() calcule tous les gradients automatiquement',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch

# ══ Backpropagation — Autograd ══

# ── 1. Calcul simple ──
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b        # Forward
loss = (y - 10) ** 2  # Perte

print(f"y = w·x + b = {y.item():.2f}")
print(f"loss = (y - 10)² = {loss.item():.2f}")

loss.backward()       # Backward

print(f"\\n∂loss/∂w = {w.grad.item():.2f}")
print(f"∂loss/∂x = {x.grad.item():.2f}")
print(f"∂loss/∂b = {b.grad.item():.2f}")

# ── 2. Graphe de calcul plus complexe ──
a = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

c = a * b           # c = 3.0
d = torch.relu(c - 2.5)  # d = ReLU(0.5) = 0.5
e = d ** 2           # e = 0.25

e.backward()
print(f"\\n── Graphe complexe ──")
print(f"∂e/∂a = {a.grad.item():.4f}")
print(f"∂e/∂b = {b.grad.item():.4f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 8 — RÉGULARISATION
  // ═══════════════════════════════════════
  {
    id: 'regularization',
    title: 'Régularisation & Généralisation',
    shortTitle: 'Régular.',
    description: 'L2, Dropout, Data Augmentation, Early Stopping — éviter le surapprentissage.',
    status: 'locked',
    progress: 0,
    dependencies: ['backprop'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `Le **surapprentissage** (overfitting) se produit quand le modèle mémorise les données d'entraînement au lieu d'apprendre des patterns généraux. La **régularisation** ajoute des contraintes pour favoriser la généralisation.\n\nSources d'erreur :\n- **Biais** : le modèle est trop simple (underfitting)\n- **Variance** : le modèle est trop sensible aux données (overfitting)\n- **Bruit** : erreur irréductible dans les données`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{reg} = \\mathcal{L}_{data} + \\lambda \\| \\boldsymbol{\\phi} \\|_2^2',
        label: 'Régularisation L2 (Weight Decay)',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `Le **Dropout** désactive aléatoirement une fraction p des neurones pendant l'entraînement, forçant le réseau à ne pas dépendre d'un seul neurone. À l'inférence, tous les neurones sont actifs mais leurs sorties sont multipliées par (1-p).\n\n**Early Stopping** : on surveille la perte de validation et on arrête l'entraînement quand elle commence à augmenter.\n\n**Data Augmentation** : on augmente artificiellement le dataset (rotations, flips, crops pour les images).`,
      },
      {
        type: 'equation',
        content: '\\tilde{h}_k = h_k \\cdot m_k \\quad \\text{où } m_k \\sim \\text{Bernoulli}(1-p)',
        label: 'Dropout (pendant l\'entraînement)',
      },
      {
        type: 'callout',
        content: '🧠 La **Batch Normalization** normalise les activations de chaque couche pour avoir une moyenne de 0 et une variance de 1. Elle agit à la fois comme régularisation et accélérateur d\'entraînement.',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Régularisation ══

# ── 1. Weight Decay (L2) ──
model = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(),
    nn.Linear(50, 1)
)

# Adam avec weight_decay = L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
print("✓ Weight decay activé (λ=0.01)")

# ── 2. Dropout ──
model_with_dropout = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Dropout(p=0.3),    # 30% des neurones désactivés
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(50, 1)
)

# En mode training vs eval
x = torch.randn(1, 10)

model_with_dropout.train()
out1 = model_with_dropout(x)
out2 = model_with_dropout(x)
print(f"\\nTrain mode (dropout actif):")
print(f"  Sortie 1: {out1.item():.4f}")
print(f"  Sortie 2: {out2.item():.4f} (différent!)")

model_with_dropout.eval()
out3 = model_with_dropout(x)
out4 = model_with_dropout(x)
print(f"\\nEval mode (dropout désactivé):")
print(f"  Sortie 1: {out3.item():.4f}")
print(f"  Sortie 2: {out4.item():.4f} (identique!)")

# ── 3. Batch Normalization ──
bn_model = nn.Sequential(
    nn.Linear(10, 50),
    nn.BatchNorm1d(50),  # Normalise les activations
    nn.ReLU(),
    nn.Linear(50, 1)
)
print(f"\\n✓ BatchNorm model: {sum(p.numel() for p in bn_model.parameters())} params")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 9 — CNN
  // ═══════════════════════════════════════
  {
    id: 'cnn',
    title: 'Réseaux Convolutifs (CNN)',
    shortTitle: 'CNN',
    description: 'Convolutions 2D, invariance, équivariance, pooling — la vision par ordinateur.',
    status: 'locked',
    progress: 0,
    dependencies: ['regularization'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `Les **CNN** exploitent une propriété clé des images : les patterns locaux (bords, textures) sont les mêmes quel que soit leur position. Deux concepts formalisent cette idée :\n\n- **Invariance** : f[t[x]] = f[x] — la sortie ne change pas sous une transformation (ex: classification)\n- **Équivariance** : f[t[x]] = t[f[x]] — la sortie se transforme de la même façon (ex: segmentation)`,
      },
      {
        type: 'equation',
        content: 'z_i = \\sum_{m} \\omega_m \\cdot x_{i+m}',
        label: 'Convolution 1D (kernel de taille M)',
      },
      {
        type: 'text',
        content: `La **convolution 2D** applique un filtre (kernel) qui glisse sur l'image. Ce filtre détecte des patterns locaux (bords, coins, textures). Les mêmes poids sont partagés partout (equivariance à la translation).\n\n**Pooling** (Max/Average) réduit la résolution spatiale et rend le réseau partiellement invariant à de petites translations.`,
      },
      {
        type: 'equation',
        content: 'z_{ij} = \\sum_{m} \\sum_{n} \\omega_{mn} \\cdot x_{i+m, \\, j+n}',
        label: 'Convolution 2D',
      },
      {
        type: 'callout',
        content: '⚡ Un CNN typique empile : Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC. Les premières couches détectent des features bas-niveau (bords), les dernières des features haut-niveau (visages, objets).',
      },
    ],
    exercises: [
      {
        id: 'cnn-ex1',
        title: 'Construire un CNN pour MNIST',
        instructions: 'Créez un CNN simple avec 2 couches convolutives pour classifier des images 28×28.',
        starterCode: `import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ___  # 1 canal → 16 filtres, kernel 3
        self.conv2 = ___  # 16 → 32 filtres, kernel 3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = ___     # 32*7*7 → 10 classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)
out = model(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")`,
        solution: `import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)
out = model(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")`,
        hints: [
          'nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)',
          'Après 2 MaxPool(2,2) sur 28×28 → 7×7',
          'nn.Linear(32 * 7 * 7, 10)',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Réseaux Convolutifs (CNN) ══

# ── 1. Convolution 2D simple ──
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(1, 1, 28, 28)  # batch=1, channels=1, H=28, W=28
out = conv(x)
print(f"Conv2d: {x.shape} → {out.shape}")
print(f"Paramètres conv: {conv.weight.shape} = {conv.weight.numel()} poids + {conv.bias.numel()} biais")

# ── 2. CNN complet ──
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → 14×14
            nn.Conv2d(16, 32, 3, padding=1),   # → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

model = MNISTNet()
out = model(torch.randn(4, 1, 28, 28))
print(f"\\nMNISTNet: batch=4 → {out.shape}")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 10 — RESIDUAL NETWORKS
  // ═══════════════════════════════════════
  {
    id: 'resnet',
    title: 'Réseaux Résiduels (ResNet)',
    shortTitle: 'ResNet',
    description: 'Connexions résiduelles, skip connections et Batch Normalization.',
    status: 'locked',
    progress: 0,
    dependencies: ['cnn'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `Les réseaux très profonds (>20 couches) souffrent du problème des **gradients évanescents** : le gradient devient exponentiellement petit en traversant chaque couche. Les **connexions résiduelles** (skip connections) résolvent ce problème en ajoutant l'entrée directement à la sortie de chaque bloc.`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}_{k+1} = \\mathbf{h}_k + f_k(\\mathbf{h}_k)',
        label: 'Connexion résiduelle',
        highlightVar: 'hidden',
      },
      {
        type: 'text',
        content: `Au lieu d'apprendre la transformation complète h → h', le réseau apprend le **résidu** f(h) = h' - h. Si le résidu est proche de zéro, le gradient passe directement via le skip connection.\n\nUn **bloc résiduel** typique contient :\n- BatchNorm → ReLU → Conv → BatchNorm → ReLU → Conv → + input`,
      },
      {
        type: 'callout',
        content: '🧠 ResNet a permis d\'entraîner des réseaux de 152+ couches. Sans skip connections, même des réseaux de 30 couches étaient difficiles à entraîner. L\'idée clé : le réseau peut toujours "copier" l\'entrée si les couches ne sont pas utiles.',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Réseaux Résiduels ══

class ResidualBlock(nn.Module):
    """Bloc résiduel : sortie = entrée + f(entrée)"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection !

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        self.res3 = ResidualBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleResNet()
x = torch.randn(2, 1, 28, 28)
out = model(x)
print(f"SimpleResNet: {x.shape} → {out.shape}")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
print(f"Profondeur: 1 conv + 6 conv (3 blocs × 2) = 7 couches conv")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 11 — RNN/LSTM
  // ═══════════════════════════════════════
  {
    id: 'rnn',
    title: 'Réseaux Récurrents (RNN/LSTM)',
    shortTitle: 'RNN',
    description: 'Traitement des séquences avec mémoire temporelle et portes.',
    status: 'locked',
    progress: 0,
    dependencies: ['regularization'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `Les **RNN** traitent des séquences en maintenant un **état caché** (hidden state) qui encode l'historique. À chaque pas de temps t, le RNN reçoit l'entrée xₜ et l'état précédent hₜ₋₁, et produit un nouvel état hₜ.\n\nLe problème : les gradients s'évanouissent sur les longues séquences (long-range dependencies).`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}_t = \\tanh(\\mathbf{W}_{hh} \\mathbf{h}_{t-1} + \\mathbf{W}_{xh} \\mathbf{x}_t + \\mathbf{b}_h)',
        label: 'RNN — État caché',
      },
      {
        type: 'text',
        content: `Le **LSTM** (Long Short-Term Memory) résout le vanishing gradient avec des **portes** (gates) qui contrôlent le flux d'information :\n\n- **Porte d'oubli** (forget gate) : quelle info effacer de la mémoire\n- **Porte d'entrée** (input gate) : quelle nouvelle info stocker\n- **Porte de sortie** (output gate) : quelle info envoyer en sortie`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{f}_t &= \\sigma(\\mathbf{W}_f [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_f) \\\\ \\mathbf{i}_t &= \\sigma(\\mathbf{W}_i [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_i) \\\\ \\mathbf{c}_t &= \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tanh(\\mathbf{W}_c [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_c) \\end{aligned}',
        label: 'LSTM — Portes et cellule mémoire',
      },
      {
        type: 'callout',
        content: '💡 Les Transformers ont largement remplacé les RNN/LSTM pour la plupart des tâches séquentielles (NLP, audio). Cependant, les RNN restent utiles pour les séquences très longues et le traitement en temps réel.',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Réseaux Récurrents ══

# ── 1. RNN simple ──
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(1, 5, 10)  # batch=1, seq_len=5, features=10
output, h_n = rnn(x)
print(f"RNN Output: {output.shape}")    # (1, 5, 20)
print(f"RNN Hidden: {h_n.shape}")       # (2, 1, 20)

# ── 2. LSTM ──
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)
print(f"\\nLSTM Output: {output.shape}")
print(f"LSTM Hidden: {h_n.shape}")
print(f"LSTM Cell:   {c_n.shape}")

# ── 3. LSTM pour classification de séquences ──
class SeqClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # Utiliser le dernier état caché
        return self.fc(h_n.squeeze(0))

clf = SeqClassifier(10, 32, 5)
x = torch.randn(4, 20, 10)  # batch=4, seq_len=20, features=10
out = clf(x)
print(f"\\nClassification: {x.shape} → {out.shape}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 12 — TRANSFORMERS
  // ═══════════════════════════════════════
  {
    id: 'attention',
    title: 'Attention & Transformers',
    shortTitle: 'Transformer',
    description: 'Self-Attention, Multi-Head Attention et l\'architecture qui a révolutionné le NLP.',
    status: 'locked',
    progress: 0,
    dependencies: ['rnn'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `Le **Transformer** est l'architecture dominante en Deep Learning moderne. Son mécanisme clé est le **dot-product self-attention** qui permet à chaque élément d'une séquence de "consulter" tous les autres.\n\nPour chaque entrée xₘ, on calcule trois vecteurs :\n- **Value** vₘ : le contenu à transmettre\n- **Query** qₙ : "quelle information cherche la position n ?"\n- **Key** kₘ : "quelle information offre la position m ?"`,
      },
      {
        type: 'equation',
        content: '\\text{sa}_n[\\mathbf{x}_\\bullet] = \\sum_{m=1}^{N} a[\\mathbf{x}_m, \\mathbf{x}_n] \\cdot \\mathbf{v}_m',
        label: 'Self-Attention Output',
      },
      {
        type: 'text',
        content: `Les poids d'attention sont calculés par produit scalaire queries × keys, divisé par √dₖ pour la stabilité numérique, puis passés par softmax. Le nombre de poids d'attention croît quadratiquement avec la longueur de séquence N.`,
      },
      {
        type: 'equation',
        content: '\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\!\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right) \\mathbf{V}',
        label: 'Scaled Dot-Product Attention',
      },
      {
        type: 'text',
        content: `Le **Multi-Head Attention** exécute H mécanismes d'attention en parallèle, chacun apprenant des types de relations différents. BERT utilise H=12 têtes, GPT-3 utilise H=96 têtes.\n\nUne couche Transformer complète = Multi-Head Attention + Add&Norm + FFN + Add&Norm.`,
      },
      {
        type: 'callout',
        content: '🧠 Le self-attention est un **hypernetwork** : une branche du réseau (Q,K) calcule les poids pour une autre branche (V). C\'est ce qui rend les Transformers si flexibles — les connexions dépendent des données elles-mêmes.',
      },
    ],
    exercises: [
      {
        id: 'attn-ex1',
        title: 'Self-Attention from scratch',
        instructions: 'Implémentez le scaled dot-product attention manuellement (sans nn.MultiheadAttention).',
        starterCode: `import torch
import torch.nn as nn
import math

# Dimensions
d_model = 64
seq_len = 5
batch = 1

# Projections Q, K, V
W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

x = torch.randn(batch, seq_len, d_model)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

# Scaled dot-product attention
scores = ___  # Q @ K^T / sqrt(d_k)
weights = ___  # softmax(scores)
output = ___   # weights @ V

print(f"Scores shape: {scores.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Output shape: {output.shape}")`,
        solution: `import torch
import torch.nn as nn
import math

d_model = 64
seq_len = 5
batch = 1

W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

x = torch.randn(batch, seq_len, d_model)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)
weights = torch.softmax(scores, dim=-1)
output = weights @ V

print(f"Scores shape: {scores.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Output shape: {output.shape}")`,
        hints: [
          'scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)',
          'weights = torch.softmax(scores, dim=-1)',
          'output = weights @ V',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn
import math

# ══ Self-Attention & Transformers ══

class SelfAttention(nn.Module):
    """Scaled Dot-Product Self-Attention avec Multi-Head"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Projections Q, K, V puis split en têtes
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        # Combiner les têtes
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

# Test
attn = SelfAttention(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
out = attn(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")
print(f"Heads:  {attn.n_heads}, d_k: {attn.d_k}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 13 — GANs
  // ═══════════════════════════════════════
  {
    id: 'gan',
    title: 'Generative Adversarial Networks (GAN)',
    shortTitle: 'GAN',
    description: 'Génération d\'images via un duel Générateur vs Discriminateur.',
    status: 'locked',
    progress: 0,
    dependencies: ['attention'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `Un **GAN** met en compétition deux réseaux :\n\n- **Générateur G** : transforme du bruit aléatoire z en données réalistes G(z)\n- **Discriminateur D** : distingue les données réelles des données générées\n\nLe générateur cherche à tromper le discriminateur. Le discriminateur cherche à ne pas être trompé. Ce jeu adversarial conduit le générateur à produire des données de plus en plus réalistes.`,
      },
      {
        type: 'equation',
        content: '\\min_G \\max_D \\; \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]',
        label: 'Objectif du GAN (Minimax)',
      },
      {
        type: 'callout',
        content: '⚡ Les GANs sont notoirement difficiles à entraîner (mode collapse, instabilité). Des variantes comme WGAN, StyleGAN, et la progressive growing ont résolu beaucoup de ces problèmes.',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Generative Adversarial Network (GAN) ══

# Générateur : bruit → image
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

# Discriminateur : image → réel/faux
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()

# Générer une image depuis du bruit
z = torch.randn(1, 100)
fake_img = G(z)
score = D(fake_img)

print(f"Bruit z: {z.shape}")
print(f"Image générée: {fake_img.shape}")
print(f"Score discriminateur: {score.item():.4f} (0=faux, 1=vrai)")
print(f"\\nG params: {sum(p.numel() for p in G.parameters()):,}")
print(f"D params: {sum(p.numel() for p in D.parameters()):,}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 14 — DIFFUSION MODELS
  // ═══════════════════════════════════════
  {
    id: 'diffusion',
    title: 'Modèles de Diffusion',
    shortTitle: 'Diffusion',
    description: 'Le processus de bruitage/débruitage qui génère des images photoréalistes.',
    status: 'locked',
    progress: 0,
    dependencies: ['attention'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `Les **modèles de diffusion** apprennent à générer des données en inversant un processus de bruitage progressif. Le modèle apprend à **débruiter** — à chaque étape, il enlève un peu de bruit pour reconstruire l'image originale.\n\n- **Forward process** (encoder) : ajouter progressivement du bruit gaussien à l'image\n- **Reverse process** (decoder) : apprendre à retirer le bruit étape par étape`,
      },
      {
        type: 'equation',
        content: 'q(\\mathbf{x}_t | \\mathbf{x}_{t-1}) = \\mathcal{N}(\\mathbf{x}_t; \\sqrt{1-\\beta_t} \\, \\mathbf{x}_{t-1}, \\beta_t \\mathbf{I})',
        label: 'Forward Process (ajout de bruit)',
      },
      {
        type: 'equation',
        content: '\\mathcal{L} = \\mathbb{E}_{t, \\mathbf{x}_0, \\boldsymbol{\\epsilon}} \\left[ \\| \\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t) \\|^2 \\right]',
        label: 'Objectif simplifié (prédire le bruit)',
        highlightVar: 'loss',
      },
      {
        type: 'callout',
        content: '🧠 DALL-E, Stable Diffusion, et Midjourney utilisent tous des modèles de diffusion. L\'idée clé : au lieu de générer une image d\'un coup, on la "débruite" progressivement depuis du bruit pur en T étapes (typiquement T=1000).',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Modèle de Diffusion — Concept simplifié ══

class SimpleDenoiser(nn.Module):
    """Réseau qui prédit le bruit ajouté à une image"""
    def __init__(self, dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),  # +1 pour le timestep t
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, x_noisy, t):
        """Prédit le bruit epsilon à partir de x_noisy et t"""
        t_embed = t.unsqueeze(-1)  # (batch, 1)
        inp = torch.cat([x_noisy, t_embed], dim=-1)
        return self.net(inp)

# ── Forward process : ajouter du bruit ──
T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def add_noise(x0, t, noise=None):
    """Ajoute du bruit au timestep t"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t]).unsqueeze(-1)
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t]).unsqueeze(-1)
    return sqrt_ab * x0 + sqrt_1_ab * noise, noise

# Test
model = SimpleDenoiser(dim=784)
x0 = torch.randn(4, 784)  # 4 images "propres"
t = torch.randint(0, T, (4,))  # timesteps aléatoires

x_noisy, true_noise = add_noise(x0, t)
pred_noise = model(x_noisy, t.float() / T)

loss = nn.MSELoss()(pred_noise, true_noise)
print(f"x0 shape: {x0.shape}")
print(f"x_noisy shape: {x_noisy.shape}")
print(f"Loss: {loss.item():.4f}")
print(f"\\nObjectif: prédire le bruit ε ajouté à l'image")
`,
  },
];

// ── Graph positions for Roadmap visualization ──
export const nodePositions: Record<string, { x: number; y: number }> = {
  // Row 1 — Fundamentals
  'supervised-learning': { x: 100, y: 80 },
  'tensors':             { x: 350, y: 80 },
  'shallow-networks':    { x: 600, y: 80 },
  'deep-networks':       { x: 850, y: 80 },
  // Row 2 — Training
  'loss-functions':      { x: 100, y: 260 },
  'gradient-descent':    { x: 350, y: 260 },
  'backprop':            { x: 600, y: 260 },
  'regularization':      { x: 850, y: 260 },
  // Row 3 — Architectures
  'cnn':                 { x: 100, y: 440 },
  'resnet':              { x: 350, y: 440 },
  'rnn':                 { x: 600, y: 440 },
  // Row 4 — Advanced
  'attention':           { x: 350, y: 620 },
  'gan':                 { x: 600, y: 620 },
  'diffusion':           { x: 850, y: 620 },
};

export function getTotalProgress(nodes: CourseNode[]): number {
  const completed = nodes.filter(n => n.status === 'completed').length;
  return Math.round((completed / nodes.length) * 100);
}
