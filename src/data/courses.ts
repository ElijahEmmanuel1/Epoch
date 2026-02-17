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
  // MODULE 4 — RÉSEAUX PROFONDS (Ch. 4)
  // ═══════════════════════════════════════
  {
    id: 'deep-networks',
    title: 'Réseaux de Neurones Profonds',
    shortTitle: 'Deep Nets',
    description: 'Composition de réseaux, profondeur vs largeur, régions linéaires exponentielles et notation matricielle (Ch. 4 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['shallow-networks'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `## 4.1 — Composition de réseaux\n\nUn **réseau profond** est obtenu en **composant** des réseaux superficiels : la sortie du premier devient l'entrée du second. Le premier réseau "plie" (**fold**) l'espace d'entrée : plusieurs valeurs de x sont mappées sur la même valeur y. Le second réseau applique sa fonction, qui est alors **dupliquée** à chaque pli.`,
      },
      {
        type: 'diagram',
        content: `  ┌─────────────┐     ┌─────────────┐
  │  Réseau 1    │     │  Réseau 2    │
  │  x → y       │────▶│  y → y'      │
  │  3 hidden    │     │  3 hidden    │
  │  4 régions   │     │  4 régions   │
  └─────────────┘     └─────────────┘
        │                    │
        ▼                    ▼
  3 "plis" du              Fonction dupliquée
  domaine x                3 × 3 = 9 régions !`,
        label: 'Fig. 4.1 — Composer 2 réseaux : pliage + duplication',
      },
      {
        type: 'callout',
        content: '🧠 **Intuition du pliage** : le premier réseau replie l\'espace d\'entrée. Le second réseau travaille sur l\'espace replié. En "dépliant", on voit que la fonction du second réseau est répliquée à chaque pli, variously flipped et rescaled.',
      },
      {
        type: 'text',
        content: `## 4.2 — De la composition au réseau profond\n\nCette composition est un **cas particulier** d'un réseau à 2 couches cachées. Le réseau général est plus expressif car les poids entre couches sont **libres** (pas contraints au produit extérieur).`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} h_d\' &= a\\!\\left[\\psi_{d0} + \\psi_{d1}h_1 + \\psi_{d2}h_2 + \\psi_{d3}h_3\\right] \\end{aligned}',
        label: 'Éq. 4.6 — Couche cachée 2 : fonction des activations de la couche 1',
        highlightVar: 'hidden',
      },
      {
        type: 'text',
        content: `## 4.3 — Réseau profond général\n\nUn réseau à K couches cachées applique alternativement des transformations affines et des activations ReLU. Le calcul procède couche par couche :\n\n1. **Pré-activations** fₖ = βₖ + Ωₖhₖ (transformation affine)\n2. **Activations** hₖ₊₁ = a[fₖ] (ReLU "clippe" les négatifs, crée de nouveaux joints)\n3. La sortie finale est une dernière combinaison linéaire`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{h}_1 &= a[\\boldsymbol{\\beta}_0 + \\boldsymbol{\\Omega}_0 \\mathbf{x}] \\\\ \\mathbf{h}_k &= a[\\boldsymbol{\\beta}_{k-1} + \\boldsymbol{\\Omega}_{k-1} \\mathbf{h}_{k-1}] \\\\ \\mathbf{y} &= \\boldsymbol{\\beta}_K + \\boldsymbol{\\Omega}_K \\mathbf{h}_K \\end{aligned}',
        label: 'Éq. 4.15 — Réseau profond à K couches cachées',
        highlightVar: 'hidden',
      },
      {
        type: 'diagram',
        content: `  x (Dᵢ)                                         y (Dₒ)
  ─┬─      Ω₀         Ω₁         Ω₂         Ω₃
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐
   ├─▶│ β₀+Ω₀·x │▶│ β₁+Ω₁·h₁│▶│ β₂+Ω₂·h₂│▶│β₃+Ω₃│──▶ y
   │  │  ReLU    │ │  ReLU    │ │  ReLU    │ │·h₃   │
   │  └──────────┘ └──────────┘ └──────────┘ └──────┘
   │     h₁(D₁)      h₂(D₂)      h₃(D₃)
   │
  Dᵢ=3, D₁=4, D₂=2, D₃=3, Dₒ=2  (ex. Fig. 4.6)`,
        label: 'Fig. 4.6 — Architecture avec K=3 couches cachées',
      },
      {
        type: 'text',
        content: `## 4.4 — Hyperparamètres\n\nLe nombre de couches **K** (profondeur) et le nombre d'unités par couche **D₁, D₂, ..., Dₖ** (largeur) sont des **hyperparamètres** : ils sont fixés *avant* l'apprentissage des poids. Pour des hyperparamètres fixés, les poids définissent une fonction particulière. En changeant les hyperparamètres, on explore une "famille de familles" de fonctions.`,
      },
      {
        type: 'text',
        content: `## 4.5 — Profondeur vs Largeur\n\n**Nombre de régions linéaires** : avec D unités par couche et K couches, le nombre maximum de régions est **(D+1)^K** (vs D+1 pour un réseau superficiel). L'explosion est exponentielle :`,
      },
      {
        type: 'diagram',
        content: `  K=1 (shallow)  │  K=2           │  K=5
  D=10            │  D=10          │  D=10
  ─────────────── │ ────────────── │ ──────────────
  11 régions      │  121 régions   │  161,051 régions
  31 params       │  141 params    │  471 params

  → Avec le MÊME budget de paramètres,
    le réseau profond crée exponentiellement plus de régions !`,
        label: 'Fig. 4.7 — Régions linéaires : shallow vs deep',
      },
      {
        type: 'callout',
        content: '⚡ **Depth efficiency** : certaines fonctions nécessitent un réseau superficiel avec **exponentiellement** plus d\'unités cachées pour atteindre la même approximation qu\'un réseau profond. C\'est pourquoi en pratique, les meilleurs résultats sont obtenus avec des dizaines ou centaines de couches.',
      },
      {
        type: 'text',
        content: `**En PyTorch**, un réseau profond se construit soit avec \`nn.Sequential\` soit avec \`nn.Module\` personnalisé :\n\n- \`nn.Sequential(*layers)\` : empile les couches, forward automatique\n- \`nn.Module\` : plus flexible, permet des skip connections (voir ResNet)\n- \`model.named_parameters()\` : inspecte couche par couche\n- \`torchsummary.summary(model, input_size)\` : résumé complet`,
      },
      {
        type: 'text',
        content: `## 4.6 — Comptage des paramètres (réseau profond)\n\nPour un réseau à K couches avec Dₖ unités par couche :`,
      },
      {
        type: 'equation',
        content: 'N_{\\text{params}} = \\sum_{k=0}^{K} D_{k+1} \\cdot (D_k + 1) \\quad \\text{où } D_0 = D_i,\\; D_{K+1} = D_o',
        label: 'Nombre de paramètres d\'un réseau profond',
      },
      {
        type: 'callout',
        content: '🧠 **Résumé Ch. 4** :\n(1) Composer des réseaux = plier l\'espace d\'entrée\n(2) Chaque couche clippe (ReLU) et crée de nouveaux joints\n(3) Régions max = (D+1)^K — croissance exponentielle\n(4) Le deep learning est efficace car les fonctions réelles sont souvent compositionnelles\n(5) Pour les grandes entrées structurées (images), le traitement local-to-global nécessite la profondeur',
      },
    ],
    exercises: [
      {
        id: 'deep-ex1',
        title: '💻 Pratique — Réseau à 3 couches cachées',
        instructions: 'Créez un réseau nn.Module avec 3 couches cachées (784→256→128→64→10). Comptez les paramètres par couche et au total.',
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

# Comptage par couche
for name, param in model.named_parameters():
    print(f"  {name:15s} : {str(list(param.shape)):15s} = {param.numel():>7,} params")

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

for name, param in model.named_parameters():
    print(f"  {name:15s} : {str(list(param.shape)):15s} = {param.numel():>7,} params")

total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal paramètres: {total:,}")`,
        hints: [
          'nn.Linear(in_features, out_features) crée une couche dense',
          'Couche 1: 784→256, Couche 2: 256→128, Couche 3: 128→64, Sortie: 64→10',
        ],
        completed: false,
      },
      {
        id: 'deep-th1',
        title: '🧠 Théorie — Régions linéaires (Prob. 4.8)',
        instructions: 'Calculez et comparez le nombre maximum de régions linéaires pour des réseaux de profondeur K=1 à K=10, avec D=10 unités par couche. Vérifiez la formule (D+1)^K.',
        starterCode: `import torch

def max_regions_shallow(D):
    """Régions max pour réseau superficiel"""
    return D + 1

def max_regions_deep(D, K):
    """Régions max pour réseau profond à K couches, D unités/couche"""
    return ___

def count_params_deep(D_i, D, K, D_o):
    """Nombre de paramètres d'un réseau profond"""
    # Couche 1 : D*(D_i+1)
    # Couches 2..K : (K-1)*D*(D+1)
    # Sortie : D_o*(D+1)
    return ___

D = 10
D_i, D_o = 1, 1

print(f"{'K':>3} │ {'Params':>8} │ {'Régions max':>15} │ {'Régions/param':>15}")
print(f"{'─'*3}─┼─{'─'*8}─┼─{'─'*15}─┼─{'─'*15}")
for K in range(1, 11):
    n_params = count_params_deep(D_i, D, K, D_o)
    n_regions = max_regions_deep(D, K)
    ratio = n_regions / n_params
    print(f"{K:3d} │ {n_params:8,} │ {n_regions:15,} │ {ratio:15.1f}")`,
        solution: `import torch

def max_regions_shallow(D):
    return D + 1

def max_regions_deep(D, K):
    return (D + 1) ** K

def count_params_deep(D_i, D, K, D_o):
    return D * (D_i + 1) + (K - 1) * D * (D + 1) + D_o * (D + 1)

D = 10
D_i, D_o = 1, 1

print(f"{'K':>3} │ {'Params':>8} │ {'Régions max':>15} │ {'Régions/param':>15}")
print(f"{'─'*3}─┼─{'─'*8}─┼─{'─'*15}─┼─{'─'*15}")
for K in range(1, 11):
    n_params = count_params_deep(D_i, D, K, D_o)
    n_regions = max_regions_deep(D, K)
    ratio = n_regions / n_params
    print(f"{K:3d} │ {n_params:8,} │ {n_regions:15,} │ {ratio:15.1f}")`,
        hints: [
          'Régions max = (D+1)^K',
          'Params couche 1 = D*(D_i+1), couches internes = D*(D+1), sortie = D_o*(D+1)',
        ],
        completed: false,
      },
      {
        id: 'deep-pr2',
        title: '💻 Pratique — Deep vs Shallow (sin approximation)',
        instructions: 'Comparez un réseau superficiel (D=100) et un réseau profond (K=3, D=20) pour approximer sin(x). Les deux ont ~300 paramètres — lequel converge le mieux ?',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y = torch.sin(x)

# Réseau SUPERFICIEL : 1 → 100 → 1
shallow = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1)
)

# Réseau PROFOND : 1 → 20 → 20 → 20 → 1
deep = nn.Sequential(
    ___,  # 4 couches à remplir
)

print(f"Shallow params: {sum(p.numel() for p in shallow.parameters())}")
print(f"Deep params:    {sum(p.numel() for p in deep.parameters())}")

def train(model, epochs=2000):
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss_fn(model(x), y).item()

loss_s = train(shallow)
loss_d = train(deep)
print(f"\\nShallow loss: {loss_s:.6f}")
print(f"Deep loss:    {loss_d:.6f}")
print(f"→ {'Deep' if loss_d < loss_s else 'Shallow'} gagne !")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y = torch.sin(x)

shallow = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1)
)

deep = nn.Sequential(
    nn.Linear(1, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 1),
)

print(f"Shallow params: {sum(p.numel() for p in shallow.parameters())}")
print(f"Deep params:    {sum(p.numel() for p in deep.parameters())}")

def train(model, epochs=2000):
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss_fn(model(x), y).item()

loss_s = train(shallow)
loss_d = train(deep)
print(f"\\nShallow loss: {loss_s:.6f}")
print(f"Deep loss:    {loss_d:.6f}")
print(f"→ {'Deep' if loss_d < loss_s else 'Shallow'} gagne !")`,
        hints: [
          'nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), ...',
          'Les deux modèles ont ~300 params mais le deep crée plus de régions',
        ],
        completed: false,
      },
      {
        id: 'deep-th2',
        title: '🧠 Théorie — Activation linéaire profonde (Prob. 4.1)',
        instructions: 'Prob. 4.1 : montrez que si on compose deux réseaux SANS activation (fonction identité au lieu de ReLU), le résultat est encore une simple fonction linéaire. Démontrez-le numériquement.',
        starterCode: `import torch

# Réseau 1 : y = phi0 + phi1*x (linéaire)
# Réseau 2 : y' = phi0' + phi1'*y (linéaire)
# Composition : y' = phi0' + phi1'*(phi0 + phi1*x)
#             = (phi0' + phi1'*phi0) + (phi1'*phi1)*x
# → Encore linéaire !

# Démonstration avec des couches PyTorch (sans ReLU)
import torch.nn as nn

# Deep network SANS activation
deep_linear = nn.Sequential(
    nn.Linear(1, 50),
    nn.Linear(50, 50),
    nn.Linear(50, 50),
    nn.Linear(50, 1),
)

# Vérifier que c'est une droite
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = deep_linear(x).detach()

# Fit linéaire : y ≈ ax + b
x_np = x.squeeze().numpy()
y_np = y.squeeze().numpy()
a = (y_np[-1] - y_np[0]) / (x_np[-1] - x_np[0])
b = y_np[0] - a * x_np[0]

# Vérifier que TOUS les points sont sur la droite
y_linear = a * x_np + b
max_error = max(abs(y_np - y_linear))
print(f"Pente a = {a:.4f}, offset b = {b:.4f}")
print(f"Erreur max vs droite : {max_error:.10f}")
print(f"→ {'✓ C\\'est bien une DROITE !' if max_error < 1e-5 else '✗ Pas linéaire'}") 
print(f"\\n💡 Sans activation non-linéaire, empiler des couches")
print(f"   n'ajoute AUCUNE expressivité. C'est pourquoi le ReLU est essentiel !")`,
        solution: `import torch
import torch.nn as nn

deep_linear = nn.Sequential(
    nn.Linear(1, 50),
    nn.Linear(50, 50),
    nn.Linear(50, 50),
    nn.Linear(50, 1),
)

x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = deep_linear(x).detach()

x_np = x.squeeze().numpy()
y_np = y.squeeze().numpy()
a = (y_np[-1] - y_np[0]) / (x_np[-1] - x_np[0])
b = y_np[0] - a * x_np[0]

y_linear = a * x_np + b
max_error = max(abs(y_np - y_linear))
print(f"Pente a = {a:.4f}, offset b = {b:.4f}")
print(f"Erreur max vs droite : {max_error:.10f}")
print(f"→ {'✓ C\\'est bien une DROITE !' if max_error < 1e-5 else '✗ Pas linéaire'}")
print(f"\\n💡 Sans activation non-linéaire, empiler des couches")
print(f"   n'ajoute AUCUNE expressivité. C'est pourquoi le ReLU est essentiel !")`,
        hints: [
          'Composition de fonctions linéaires = fonction linéaire',
          'nn.Linear sans activation entre les couches',
          'La sortie sera toujours une droite y = ax + b',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Réseaux Profonds — Ch. 4 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Réseau SUPERFICIEL vs PROFOND ──
shallow = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1)
)

deep = nn.Sequential(
    nn.Linear(1, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 1)
)

print(f"Shallow: {sum(p.numel() for p in shallow.parameters())} params")
print(f"Deep:    {sum(p.numel() for p in deep.parameters())} params")

# ── 2. Forward pass couche par couche ──
x = torch.randn(5, 1)
print(f"\\n── Forward pass détaillé ──")
h = x
for i, layer in enumerate(deep):
    h = layer(h)
    print(f"  Couche {i}: {layer.__class__.__name__:10s} → shape {list(h.shape)}")

# ── 3. Régions linéaires max ──
print(f"\\n── Régions linéaires ──")
D = 20
for K in [1, 2, 3, 5, 10]:
    regions = (D + 1) ** K
    print(f"  K={K:2d}, D={D}: {regions:>15,} régions max")

# ── 4. nn.Module personnalisé ──
class FlexibleDeepNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

model = FlexibleDeepNet([1, 50, 50, 50, 1])
print(f"\\nFlexible: {sum(p.numel() for p in model.parameters())} params")
print(f"Output: {model(torch.tensor([[1.0]])).item():.4f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 5 — FONCTIONS DE PERTE (Ch. 5)
  // ═══════════════════════════════════════
  {
    id: 'loss-functions',
    title: 'Fonctions de Perte (Loss Functions)',
    shortTitle: 'Loss',
    description: 'Maximum de vraisemblance, MSE, Binary/Multi-class Cross-Entropy, régression hétéroscédastique (Ch. 5 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['deep-networks'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `## 5.1 — La recette du Maximum de Vraisemblance\n\nLa **fonction de perte** mesure l'écart entre prédictions et réalité. Le framework universel est le **maximum de vraisemblance** en 4 étapes :\n\n1. **Choisir un modèle** de distribution Pr(y | f[x, ϕ])\n2. **Écrire la vraisemblance** L = Πᵢ Pr(yᵢ | f[xᵢ, ϕ])\n3. **Prendre le log négatif** : −log L = −Σᵢ log Pr(yᵢ | ...)\n4. **Minimiser** la perte L̂ pour trouver ϕ̂`,
      },
      {
        type: 'equation',
        content: '\\hat{\\boldsymbol{\\phi}} = \\underset{\\boldsymbol{\\phi}}{\\text{argmin}} \\left[ -\\sum_{i=1}^{I} \\log\\!\\left[ Pr(y_i \\,|\\, f[\\mathbf{x}_i, \\boldsymbol{\\phi}]) \\right] \\right]',
        label: 'Éq. 5.2 — Estimateur du maximum de vraisemblance',
      },
      {
        type: 'callout',
        content: '🧠 **Pourquoi le log ?** Le produit de probabilités → somme de logs (plus stable numériquement). La maximisation → minimisation du négatif. Le résultat est la **negative log-likelihood** (NLL).',
      },
      {
        type: 'text',
        content: `## 5.2 — Régression → MSE (Gaussienne)\n\nSi on suppose que y suit une loi **normale** centrée sur la prédiction du réseau f[x, ϕ] avec variance σ² :\n\nPr(y | f) = Normal_y[f, σ²]\n\nAlors le log négatif donne :\n−log Pr = (y − f)² / 2σ² + constante\n\nEn ignorant la constante et σ² fixe, on retrouve le **MSE** :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{\\text{MSE}} = \\frac{1}{I}\\sum_{i=1}^{I}\\left(y_i - f[\\mathbf{x}_i, \\boldsymbol{\\phi}]\\right)^2',
        label: 'Éq. 5.6 — MSE dérivé de la vraisemblance gaussienne',
        highlightVar: 'loss',
      },
      {
        type: 'diagram',
        content: `  Distribution de y|x        Dérivation
  ─────────────────────      ─────────────────────────
  y ~ Normal(f[x,ϕ], σ²)    Pr(y|x) = N(y; f, σ²)
                             log Pr = -(y-f)²/(2σ²) + C
      ╭──╮                   -log Pr ∝ (y - f)²
    ╭─╯  ╰─╮                
  ──╯ f[x,ϕ]╰──  ← σ →      → MSE Loss ! ✓`,
        label: 'Fig. 5.3 — Gaussienne → MSE',
      },
      {
        type: 'text',
        content: `## 5.3 — Classification binaire → BCE (Bernoulli)\n\nPour y ∈ {0, 1}, on modélise Pr(y=1|x) via la **sigmoïde** σ(f) = 1/(1+e^{-f}). La distribution est **Bernoulli** et la NLL donne la **Binary Cross-Entropy** :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{\\text{BCE}} = -\\frac{1}{I}\\sum_{i=1}^{I}\\left[ y_i \\log\\sigma(f_i) + (1-y_i)\\log(1-\\sigma(f_i)) \\right]',
        label: 'Éq. 5.12 — Binary Cross-Entropy',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `## 5.4 — Classification multi-classe → Softmax + CE (Catégorielle)\n\nPour K classes, le réseau produit K logits. Le **softmax** transforme ces logits en probabilités positives qui somment à 1. La perte est la **cross-entropie catégorielle** :`,
      },
      {
        type: 'equation',
        content: '\\text{softmax}_k = \\frac{e^{f_k}}{\\sum_{j=1}^{K} e^{f_j}} \\qquad\\qquad \\mathcal{L}_{\\text{CE}} = -\\sum_{i=1}^{I} \\log\\!\\left( \\text{softmax}_{y_i}(\\mathbf{f}_i) \\right)',
        label: 'Éq. 5.17/5.22 — Softmax + Cross-Entropy catégorielle',
        highlightVar: 'loss',
      },
      {
        type: 'diagram',
        content: `  Logits (sortie réseau)         Softmax            Loss
  ─────────────────────         ───────────        ──────────
  f₁ =  2.0  ─────────╲        P(c=1) = 0.659    si y=1:
  f₂ =  1.0  ──────────╋──▶    P(c=2) = 0.242    L = -log(0.659)
  f₃ =  0.1  ─────────╱        P(c=3) = 0.099       = 0.417
                        Σ=1.0 ✓

  ⚠️ PyTorch nn.CrossEntropyLoss prend les LOGITS,
     pas les probabilités ! Le softmax est inclus.`,
        label: 'Fig. — Pipeline softmax → cross-entropy',
      },
      {
        type: 'text',
        content: `## 5.5 — Régression hétéroscédastique\n\nLe MSE standard suppose un bruit **constant** σ². Mais en réalité, l'incertitude peut varier selon x. Le réseau peut prédire **deux sorties** : la moyenne μ(x) ET la variance σ²(x). La perte devient :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{\\text{hetero}} = \\sum_{i=1}^{I}\\left[ \\frac{(y_i - \\mu_i)^2}{2\\sigma_i^2} + \\frac{1}{2}\\log\\sigma_i^2 \\right]',
        label: 'Éq. 5.8 — Perte hétéroscédastique',
        highlightVar: 'loss',
      },
      {
        type: 'callout',
        content: '⚡ **Résumé des loss functions** :\n• Régression → Normal → **MSE** (\\`nn.MSELoss\\`)\n• Binaire → Bernoulli+sigmoid → **BCE** (\\`nn.BCEWithLogitsLoss\\`)\n• Multi-classe → Catégorielle+softmax → **CE** (\\`nn.CrossEntropyLoss\\`)\n• Incertitude variable → Hétéroscédastique (custom)\n\nCe sont TOUTES des cas particuliers de la NLL !',
      },
      {
        type: 'text',
        content: `**En PyTorch**, les fonctions de perte sont dans \`torch.nn\` :\n\n- \`nn.MSELoss()\` : régression\n- \`nn.BCEWithLogitsLoss()\` : binaire (inclut sigmoid, plus stable)\n- \`nn.CrossEntropyLoss()\` : multi-classe (inclut softmax)\n- \`nn.NLLLoss()\` : NLL brute (si softmax déjà appliqué via \`nn.LogSoftmax\`)\n\n⚠️ \`nn.CrossEntropyLoss\` attend des **logits** (pas des probabilités) et des **labels entiers** (pas one-hot).`,
      },
    ],
    exercises: [
      {
        id: 'loss-ex1',
        title: '💻 Pratique — MSE manuelle vs PyTorch',
        instructions: 'Implémentez le MSE manuellement et vérifiez qu\'il correspond à nn.MSELoss. Calculez aussi la NLL gaussienne complète.',
        starterCode: `import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 3.2, 4.1, 1.8])
targets = torch.tensor([3.0, 3.0, 4.0, 2.0])

# 1. MSE manuelle
mse_manual = ___  # torch.mean((pred - target)²)

# 2. MSE PyTorch
mse_pytorch = nn.MSELoss()(predictions, targets)

# 3. NLL gaussienne complète (avec sigma=0.5)
sigma = 0.5
nll = torch.mean(
    (targets - predictions)**2 / (2 * sigma**2) + torch.log(torch.tensor(sigma))
)

print(f"MSE manuelle:  {mse_manual.item():.6f}")
print(f"MSE PyTorch:   {mse_pytorch.item():.6f}")
print(f"NLL (σ={sigma}): {nll.item():.6f}")
print(f"\\n→ MSE = NLL × 2σ² = {nll.item() * 2 * sigma**2:.6f}")`,
        solution: `import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 3.2, 4.1, 1.8])
targets = torch.tensor([3.0, 3.0, 4.0, 2.0])

mse_manual = torch.mean((predictions - targets) ** 2)
mse_pytorch = nn.MSELoss()(predictions, targets)

sigma = 0.5
nll = torch.mean(
    (targets - predictions)**2 / (2 * sigma**2) + torch.log(torch.tensor(sigma))
)

print(f"MSE manuelle:  {mse_manual.item():.6f}")
print(f"MSE PyTorch:   {mse_pytorch.item():.6f}")
print(f"NLL (σ={sigma}): {nll.item():.6f}")
print(f"\\n→ MSE = NLL × 2σ² = {nll.item() * 2 * sigma**2:.6f}")`,
        hints: [
          'mse_manual = torch.mean((predictions - targets) ** 2)',
          'La NLL gaussienne = (y-f)²/(2σ²) + log(σ)',
        ],
        completed: false,
      },
      {
        id: 'loss-ex2',
        title: '💻 Pratique — BCE avec Sigmoid',
        instructions: 'Comparez nn.BCELoss (attend des probabilités) et nn.BCEWithLogitsLoss (attend des logits). Vérifiez qu\'ils donnent le même résultat.',
        starterCode: `import torch
import torch.nn as nn

logits = torch.tensor([2.0, -1.0, 0.5, 3.0])
targets = torch.tensor([1.0, 0.0, 1.0, 1.0])

# 1. Avec BCEWithLogitsLoss (recommandé)
loss_logits = nn.BCEWithLogitsLoss()(logits, targets)

# 2. Avec BCELoss (appliquer sigmoid d'abord)
probs = ___  # torch.sigmoid(logits)
loss_probs = nn.BCELoss()(probs, targets)

# 3. Manuelle
bce_manual = -torch.mean(
    targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8)
)

print(f"BCEWithLogitsLoss: {loss_logits.item():.6f}")
print(f"BCELoss (sigmoid): {loss_probs.item():.6f}")
print(f"BCE manuelle:      {bce_manual.item():.6f}")
print(f"\\n✓ Identiques !" if abs(loss_logits.item() - loss_probs.item()) < 1e-5 else "✗ Différents")`,
        solution: `import torch
import torch.nn as nn

logits = torch.tensor([2.0, -1.0, 0.5, 3.0])
targets = torch.tensor([1.0, 0.0, 1.0, 1.0])

loss_logits = nn.BCEWithLogitsLoss()(logits, targets)

probs = torch.sigmoid(logits)
loss_probs = nn.BCELoss()(probs, targets)

bce_manual = -torch.mean(
    targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8)
)

print(f"BCEWithLogitsLoss: {loss_logits.item():.6f}")
print(f"BCELoss (sigmoid): {loss_probs.item():.6f}")
print(f"BCE manuelle:      {bce_manual.item():.6f}")
print(f"\\n✓ Identiques !" if abs(loss_logits.item() - loss_probs.item()) < 1e-5 else "✗ Différents")`,
        hints: [
          'probs = torch.sigmoid(logits)',
          'BCEWithLogitsLoss = sigmoid + BCELoss en une seule opération',
        ],
        completed: false,
      },
      {
        id: 'loss-th1',
        title: '🧠 Théorie — Softmax + CE multi-classe',
        instructions: 'Implémentez softmax et cross-entropy manuellement. Vérifiez contre nn.CrossEntropyLoss. Montrez que le softmax est invariant par translation.',
        starterCode: `import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])

# 1. Softmax manuelle
def softmax_manual(z):
    e = torch.exp(z - z.max(dim=-1, keepdim=True).values)  # stabilité
    return e / e.sum(dim=-1, keepdim=True)

probs = softmax_manual(logits)
print(f"Softmax: {probs}")
print(f"Somme:   {probs.sum(dim=-1)}")

# 2. Cross-entropy manuelle
def cross_entropy_manual(logits, labels):
    probs = softmax_manual(logits)
    log_probs = torch.log(probs + 1e-8)
    return ___  # NLL

ce_manual = cross_entropy_manual(logits, labels)
ce_pytorch = nn.CrossEntropyLoss()(logits, labels)

print(f"\\nCE manuelle: {ce_manual.item():.6f}")
print(f"CE PyTorch:  {ce_pytorch.item():.6f}")

# 3. Invariance par translation
shifted = logits + 100  
print(f"\\nSoftmax invariant ? {torch.allclose(softmax_manual(logits), softmax_manual(shifted))}")`,
        solution: `import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])

def softmax_manual(z):
    e = torch.exp(z - z.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)

probs = softmax_manual(logits)
print(f"Softmax: {probs}")
print(f"Somme:   {probs.sum(dim=-1)}")

def cross_entropy_manual(logits, labels):
    probs = softmax_manual(logits)
    log_probs = torch.log(probs + 1e-8)
    return -torch.mean(log_probs[range(len(labels)), labels])

ce_manual = cross_entropy_manual(logits, labels)
ce_pytorch = nn.CrossEntropyLoss()(logits, labels)

print(f"\\nCE manuelle: {ce_manual.item():.6f}")
print(f"CE PyTorch:  {ce_pytorch.item():.6f}")

shifted = logits + 100
print(f"\\nSoftmax invariant ? {torch.allclose(softmax_manual(logits), softmax_manual(shifted))}")`,
        hints: [
          'NLL = -mean(log_probs[range(N), labels])',
          'Soustraire le max pour la stabilité numérique (log-sum-exp trick)',
        ],
        completed: false,
      },
      {
        id: 'loss-pr3',
        title: '💻 Pratique — Régression hétéroscédastique',
        instructions: 'Construisez un réseau qui prédit à la fois la moyenne μ(x) et la variance σ²(x), puis entraînez-le sur des données avec un bruit variable.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

# Données avec bruit VARIABLE
x = torch.linspace(-3, 3, 300).unsqueeze(1)
noise_std = 0.1 + 0.5 * torch.abs(x)  # bruit croissant avec |x|
y = torch.sin(x) + noise_std * torch.randn_like(x)

class HeteroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.head_mu = nn.Linear(64, 1)     # prédit μ
        self.head_logvar = nn.Linear(64, 1)  # prédit log(σ²)
    
    def forward(self, x):
        h = self.shared(x)
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        return mu, log_var

model = HeteroNet()
opt = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(1500):
    mu, log_var = model(x)
    # Perte hétéroscédastique : (y-μ)²/(2σ²) + log(σ²)/2
    loss = torch.mean(
        ___  # à compléter
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

mu, log_var = model(x)
sigma = torch.exp(0.5 * log_var).detach()
print(f"\\nσ moyen à x=0: {sigma[150].item():.3f}")
print(f"σ moyen à x=3: {sigma[-1].item():.3f}")
print(f"→ Le modèle a appris que le bruit augmente avec |x| !")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

x = torch.linspace(-3, 3, 300).unsqueeze(1)
noise_std = 0.1 + 0.5 * torch.abs(x)
y = torch.sin(x) + noise_std * torch.randn_like(x)

class HeteroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.head_mu = nn.Linear(64, 1)
        self.head_logvar = nn.Linear(64, 1)
    
    def forward(self, x):
        h = self.shared(x)
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        return mu, log_var

model = HeteroNet()
opt = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(1500):
    mu, log_var = model(x)
    loss = torch.mean(
        (y - mu)**2 / (2 * torch.exp(log_var)) + 0.5 * log_var
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

mu, log_var = model(x)
sigma = torch.exp(0.5 * log_var).detach()
print(f"\\nσ moyen à x=0: {sigma[150].item():.3f}")
print(f"σ moyen à x=3: {sigma[-1].item():.3f}")
print(f"→ Le modèle a appris que le bruit augmente avec |x| !")`,
        hints: [
          'loss = (y - mu)**2 / (2 * exp(log_var)) + 0.5 * log_var',
          'On utilise log_var au lieu de σ² pour garantir la positivité',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Fonctions de Perte — Ch. 5 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. MSE (Régression) ──
pred = torch.tensor([2.5, 3.2, 4.1, 1.8])
target = torch.tensor([3.0, 3.0, 4.0, 2.0])

print("═══ MSE (Gaussienne → Distance quadratique) ═══")
mse = nn.MSELoss()(pred, target)
mse_manual = torch.mean((pred - target) ** 2)
print(f"  PyTorch: {mse.item():.6f}")
print(f"  Manuel:  {mse_manual.item():.6f}")

# ── 2. BCE (Classification binaire) ──
print("\\n═══ BCE (Bernoulli → Binary Cross-Entropy) ═══")
logits = torch.tensor([2.0, -1.0, 0.5])
labels = torch.tensor([1.0, 0.0, 1.0])
bce = nn.BCEWithLogitsLoss()(logits, labels)
print(f"  BCE (logits): {bce.item():.6f}")

# ── 3. CE (Classification multi-classe) ──
print("\\n═══ CE (Catégorielle → Cross-Entropy) ═══")
logits_mc = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 0.3]])
labels_mc = torch.tensor([0, 1])
ce = nn.CrossEntropyLoss()(logits_mc, labels_mc)
print(f"  CE: {ce.item():.6f}")
probs = torch.softmax(logits_mc, dim=1)
print(f"  Softmax → {probs[0].tolist()}")

# ── 4. Résumé ──
print("\\n═══ Tableau récapitulatif ═══")
print("  Tâche          │ Distribution  │ Loss      │ PyTorch")
print("  ───────────────┼───────────────┼───────────┼──────────────────")
print("  Régression     │ Gaussienne    │ MSE       │ nn.MSELoss")
print("  Binaire        │ Bernoulli     │ BCE       │ nn.BCEWithLogitsLoss")
print("  Multi-classe   │ Catégorielle  │ CE        │ nn.CrossEntropyLoss")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 6 — DESCENTE DE GRADIENT (Ch. 6)
  // ═══════════════════════════════════════
  {
    id: 'gradient-descent',
    title: 'Descente de Gradient & Optimisation',
    shortTitle: 'Gradient',
    description: 'GD, SGD avec mini-batches, Momentum, Nesterov, Adam — comment entraîner un réseau (Ch. 6 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['loss-functions'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `## 6.1 — Descente de gradient (GD)\n\nLa **descente de gradient** est l'algorithme itératif standard pour minimiser la perte L[ϕ]. On part de paramètres initiaux ϕ₀ aléatoires, puis on répète :\n\n1. **Calculer le gradient** ∂L/∂ϕ (direction de plus forte montée)\n2. **Faire un pas** dans la direction **opposée** (descente)`,
      },
      {
        type: 'equation',
        content: '\\boldsymbol{\\phi}_{t+1} \\leftarrow \\boldsymbol{\\phi}_t - \\alpha \\cdot \\frac{\\partial \\mathcal{L}[\\boldsymbol{\\phi}_t]}{\\partial \\boldsymbol{\\phi}}',
        label: 'Éq. 6.3 — Mise à jour Gradient Descent',
        highlightVar: 'grad',
      },
      {
        type: 'diagram',
        content: `  Paysage de perte L(ϕ)
  ──────────────────────────────────────────
  L ↑
    │    ╭╮                  α trop grand
    │   ╱  ╲    ╭──╮         → oscillation
    │  ╱    ╲  ╱    ╲
    │ ╱      ╲╱      ╲       α juste
    │╱  ← pas ←       ╲     → convergence
    │  ─────▶ϕ*         ╲
    └────────────────────── ϕ →
        minimum local`,
        label: 'Fig. 6.3 — Learning rate α trop grand vs juste',
      },
      {
        type: 'text',
        content: `## 6.2 — SGD avec mini-batches\n\nLe GD classique calcule le gradient sur **tout** le dataset → coûteux. Le **SGD** (Stochastic Gradient Descent) utilise un **mini-batch** B de taille b à chaque itération :\n\n- Un passage complet sur toutes les données = une **epoch**\n- Le gradient est bruité mais **non-biaisé** en espérance\n- En pratique : b = 32, 64, 128, 256`,
      },
      {
        type: 'equation',
        content: '\\boldsymbol{\\phi}_{t+1} \\leftarrow \\boldsymbol{\\phi}_t - \\alpha \\cdot \\frac{1}{|\\mathcal{B}_t|} \\sum_{i \\in \\mathcal{B}_t} \\frac{\\partial \\ell_i}{\\partial \\boldsymbol{\\phi}}',
        label: 'Éq. 6.10 — SGD avec mini-batch',
        highlightVar: 'grad',
      },
      {
        type: 'text',
        content: `## 6.3 — Momentum\n\nLe SGD pur oscille dans les vallées étroites. Le **Momentum** ajoute de l'inertie : on accumule une **moyenne mobile** des gradients passés. Cela lisse la trajectoire et accélère dans les directions constantes :`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{m}_{t+1} &= \\beta \\, \\mathbf{m}_t + (1 - \\beta) \\frac{\\partial \\mathcal{L}}{\\partial \\boldsymbol{\\phi}} \\\\ \\boldsymbol{\\phi}_{t+1} &= \\boldsymbol{\\phi}_t - \\alpha \\, \\mathbf{m}_{t+1} \\end{aligned}',
        label: 'Éq. 6.11 — SGD avec Momentum (β ≈ 0.9)',
      },
      {
        type: 'text',
        content: `## 6.4 — Adam (Adaptive Moment Estimation)\n\n**Adam** combine Momentum (moyenne mobile du gradient m) et **RMSProp** (moyenne mobile du gradient²). Il adapte le learning rate **par paramètre** — les paramètres peu mis à jour reçoivent des pas plus grands :`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{m}_t &= \\beta_1 \\mathbf{m}_{t-1} + (1-\\beta_1) \\mathbf{g}_t & \\text{(1er moment)} \\\\ \\mathbf{v}_t &= \\beta_2 \\mathbf{v}_{t-1} + (1-\\beta_2) \\mathbf{g}_t^2 & \\text{(2e moment)} \\\\ \\hat{\\mathbf{m}}_t &= \\frac{\\mathbf{m}_t}{1 - \\beta_1^t} \\;,\\; \\hat{\\mathbf{v}}_t = \\frac{\\mathbf{v}_t}{1 - \\beta_2^t} & \\text{(correction biais)} \\\\ \\boldsymbol{\\phi}_t &= \\boldsymbol{\\phi}_{t-1} - \\alpha \\frac{\\hat{\\mathbf{m}}_t}{\\sqrt{\\hat{\\mathbf{v}}_t} + \\epsilon} & \\text{(mise à jour)} \\end{aligned}',
        label: 'Éq. 6.15–6.18 — Algorithme Adam',
      },
      {
        type: 'diagram',
        content: `  Comparaison des optimiseurs
  ─────────────────────────────────────────
                        Batch GD   SGD    Momentum  Adam
  ─────────────────────────────────────────
  Vitesse par step      Lent      Rapide  Rapide    Rapide
  Stabilité             +++       +       ++        +++
  Learning rate adaptatif  Non   Non     Non       OUI
  Biais-correction         -      -       -        OUI
  Usage mémoire           1×      1×      2×        3×
  Défaut α                 -     0.01    0.01     0.001
  ─────────────────────────────────────────
  β₁ = 0.9,  β₂ = 0.999,  ε = 10⁻⁸ (défauts Adam)`,
        label: 'Tableau — Comparaison des optimiseurs',
      },
      {
        type: 'text',
        content: `## 6.5 — Hyperparamètres d'entraînement\n\n- **Learning rate α** : le PLUS important. Trop grand → diverge. Trop petit → trop lent.\n- **Batch size** : grand → gradient stable mais moins de mises à jour/epoch\n- **Epochs** : nombre de passes sur les données\n- **Learning rate schedule** : réduire α au fil du temps (step decay, cosine annealing, warmup)\n\n**Recherche d'hyperparamètres** : grid search, random search (souvent meilleur), ou Bayesian optimization.`,
      },
      {
        type: 'callout',
        content: '💡 **Recette pratique** :\n1. Commencer avec **Adam** (α=0.001, β₁=0.9, β₂=0.999)\n2. Batch size = 32 ou 64\n3. Surveiller la **perte de validation** (early stopping)\n4. Si sous-optimal → essayer SGD+Momentum avec learning rate schedule\n5. Ne jamais tuner sur les données de test !',
      },
    ],
    exercises: [
      {
        id: 'gd-ex1',
        title: '💻 Pratique — SGD vs Adam',
        instructions: 'Entraînez le même modèle avec SGD, SGD+Momentum et Adam. Comparez la vitesse de convergence sur 200 epochs.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x = torch.randn(100, 1)
y = 3 * x + 2 + torch.randn(100, 1) * 0.1

loss_fn = nn.MSELoss()

# 3 copies du même modèle
model_sgd = nn.Linear(1, 1)
model_mom = nn.Linear(1, 1)
model_adam = nn.Linear(1, 1)

# Copier les mêmes poids initiaux
model_mom.load_state_dict(model_sgd.state_dict())
model_adam.load_state_dict(model_sgd.state_dict())

opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
opt_mom = optim.SGD(model_mom.parameters(), lr=0.01, momentum=___)
opt_adam = optim.Adam(model_adam.parameters(), lr=___)

for epoch in range(200):
    for model, opt, name in [(model_sgd, opt_sgd, 'SGD'),
                              (model_mom, opt_mom, 'Mom'),
                              (model_adam, opt_adam, 'Adam')]:
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    if (epoch+1) % 50 == 0:
        l1 = loss_fn(model_sgd(x), y).item()
        l2 = loss_fn(model_mom(x), y).item()
        l3 = loss_fn(model_adam(x), y).item()
        print(f"Epoch {epoch+1:3d}: SGD={l1:.4f}  Mom={l2:.4f}  Adam={l3:.4f}")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x = torch.randn(100, 1)
y = 3 * x + 2 + torch.randn(100, 1) * 0.1

loss_fn = nn.MSELoss()

model_sgd = nn.Linear(1, 1)
model_mom = nn.Linear(1, 1)
model_adam = nn.Linear(1, 1)

model_mom.load_state_dict(model_sgd.state_dict())
model_adam.load_state_dict(model_sgd.state_dict())

opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
opt_mom = optim.SGD(model_mom.parameters(), lr=0.01, momentum=0.9)
opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)

for epoch in range(200):
    for model, opt, name in [(model_sgd, opt_sgd, 'SGD'),
                              (model_mom, opt_mom, 'Mom'),
                              (model_adam, opt_adam, 'Adam')]:
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    if (epoch+1) % 50 == 0:
        l1 = loss_fn(model_sgd(x), y).item()
        l2 = loss_fn(model_mom(x), y).item()
        l3 = loss_fn(model_adam(x), y).item()
        print(f"Epoch {epoch+1:3d}: SGD={l1:.4f}  Mom={l2:.4f}  Adam={l3:.4f}")`,
        hints: [
          'momentum=0.9 pour SGD avec Momentum',
          'lr=0.001 pour Adam (learning rate adaptatif, donc plus petit)',
        ],
        completed: false,
      },
      {
        id: 'gd-th1',
        title: '🧠 Théorie — Adam from scratch',
        instructions: 'Implémentez l\'algorithme Adam manuellement (sans optim.Adam) et vérifiez la convergence sur une fonction simple f(x) = (x-3)².',
        starterCode: `import torch

# Minimiser f(x) = (x - 3)²
x = torch.tensor(10.0, requires_grad=True)

# Hyperparamètres Adam
alpha = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8
m, v = 0.0, 0.0

print(f"{'t':>3} │ {'x':>8} │ {'grad':>8} │ {'m_hat':>8} │ {'v_hat':>10} │ {'f(x)':>8}")
print("─" * 60)

for t in range(1, 51):
    # Forward
    f = (x - 3) ** 2
    f.backward()
    g = x.grad.item()
    
    # Adam update
    m = ___  # β₁ * m + (1-β₁) * g
    v = ___  # β₂ * v + (1-β₂) * g²
    m_hat = m / (1 - beta1**t)  # correction biais
    v_hat = v / (1 - beta2**t)
    
    with torch.no_grad():
        x -= alpha * m_hat / (v_hat**0.5 + eps)
        x.grad.zero_()
    
    if t <= 5 or t % 10 == 0:
        print(f"{t:3d} │ {x.item():8.4f} │ {g:8.4f} │ {m_hat:8.4f} │ {v_hat:10.6f} │ {(x.item()-3)**2:8.4f}")

print(f"\\n✓ x final = {x.item():.6f} (cible = 3.0)")`,
        solution: `import torch

x = torch.tensor(10.0, requires_grad=True)

alpha = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8
m, v = 0.0, 0.0

print(f"{'t':>3} │ {'x':>8} │ {'grad':>8} │ {'m_hat':>8} │ {'v_hat':>10} │ {'f(x)':>8}")
print("─" * 60)

for t in range(1, 51):
    f = (x - 3) ** 2
    f.backward()
    g = x.grad.item()
    
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    with torch.no_grad():
        x -= alpha * m_hat / (v_hat**0.5 + eps)
        x.grad.zero_()
    
    if t <= 5 or t % 10 == 0:
        print(f"{t:3d} │ {x.item():8.4f} │ {g:8.4f} │ {m_hat:8.4f} │ {v_hat:10.6f} │ {(x.item()-3)**2:8.4f}")

print(f"\\n✓ x final = {x.item():.6f} (cible = 3.0)")`,
        hints: [
          'm = β₁ * m + (1 - β₁) * g',
          'v = β₂ * v + (1 - β₂) * g²',
          'La correction de biais divise par (1 - βᵢᵗ)',
        ],
        completed: false,
      },
      {
        id: 'gd-pr2',
        title: '💻 Pratique — Mini-batch SGD avec DataLoader',
        instructions: 'Utilisez torch.utils.data.DataLoader pour créer des mini-batches et entraîner un réseau avec SGD.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# Dataset
X = torch.randn(1000, 5)
W_true = torch.tensor([1.0, -2.0, 3.0, -1.0, 0.5])
y = X @ W_true + 0.1 * torch.randn(1000)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=___, shuffle=True)

model = nn.Linear(5, 1, bias=False)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()

for epoch in range(20):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x).squeeze()
        loss = loss_fn(pred, batch_y)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        avg_loss = epoch_loss / len(loader)
        w = model.weight.data.squeeze()
        print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}  w={w.tolist()}")

print(f"\\nAppris: {model.weight.data.squeeze().tolist()}")
print(f"Réel:   {W_true.tolist()}")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

X = torch.randn(1000, 5)
W_true = torch.tensor([1.0, -2.0, 3.0, -1.0, 0.5])
y = X @ W_true + 0.1 * torch.randn(1000)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Linear(5, 1, bias=False)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()

for epoch in range(20):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x).squeeze()
        loss = loss_fn(pred, batch_y)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        avg_loss = epoch_loss / len(loader)
        w = model.weight.data.squeeze()
        print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}  w={w.tolist()}")

print(f"\\nAppris: {model.weight.data.squeeze().tolist()}")
print(f"Réel:   {W_true.tolist()}")`,
        hints: [
          'batch_size=32 est un bon défaut',
          'DataLoader gère le shuffling et le batching automatiquement',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn
import torch.optim as optim

# ══════════════════════════════════════════════════════════════
# Descente de Gradient — Ch. 6 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

torch.manual_seed(42)

x = torch.randn(200, 1)
y = 3 * x + 2 + torch.randn(200, 1) * 0.1

# ── 1. Les 3 optimiseurs principaux ──
model = nn.Linear(1, 1)
print("═══ Optimiseurs PyTorch ═══")

optimizers = {
    'SGD':      optim.SGD(model.parameters(), lr=0.01),
    'Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam':     optim.Adam(model.parameters(), lr=0.001),
}

for name, opt in optimizers.items():
    print(f"  {name}: {opt.__class__.__name__}")

# ── 2. Boucle d'entraînement typique ──
print("\\n═══ Entraînement avec Adam ═══")
model = nn.Linear(1, 1)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (epoch + 1) % 40 == 0:
        w, b = model.weight.item(), model.bias.item()
        print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, y={w:.3f}x + {b:.3f}")

w, b = model.weight.item(), model.bias.item()
print(f"\\n✓ Appris : y = {w:.2f}x + {b:.2f}")
print(f"  Réel  : y = 3.00x + 2.00")

# ── 3. Learning Rate Schedule ──
print("\\n═══ Learning Rate Schedule ═══")
model2 = nn.Linear(1, 1)
opt2 = optim.Adam(model2.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(opt2, step_size=50, gamma=0.5)
for epoch in range(200):
    loss = loss_fn(model2(x), y)
    opt2.zero_grad(); loss.backward(); opt2.step()
    scheduler.step()
    if (epoch+1) % 50 == 0:
        print(f"  Epoch {epoch+1}: lr = {scheduler.get_last_lr()[0]:.6f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 7 — BACKPROPAGATION (Ch. 7)
  // ═══════════════════════════════════════
  {
    id: 'backprop',
    title: 'Backpropagation & Autograd',
    shortTitle: 'Backprop',
    description: 'Règle de la chaîne, forward/backward pass, He initialization, vanishing gradients (Ch. 7 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['gradient-descent'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `## 7.1 — Forward pass & Backward pass\n\nLa **backpropagation** calcule efficacement les gradients ∂L/∂ϕ pour tous les paramètres du réseau. Le processus :\n\n1. **Forward pass** : on calcule séquentiellement f₀ → h₁ → f₁ → h₂ → ... → fₖ → perte ℓ\n2. **Backward pass** : on propage ∂ℓ/∂fₖ de la sortie vers l'entrée via la **règle de la chaîne**`,
      },
      {
        type: 'diagram',
        content: `  FORWARD  →  →  →  →  →  →  →  →  →  →  →  →
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x ──▶ f₀ ──▶ h₁ ──▶ f₁ ──▶ h₂ ──▶ f₂ ──▶ ℓ
        β₀,Ω₀  ReLU   β₁,Ω₁  ReLU   β₂,Ω₂
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ←  ←  ←  ←  ←  ←  ←  ←  ←  ←  ←  ←  BACKWARD
  ∂ℓ/∂x  ∂ℓ/∂f₀  ∂ℓ/∂h₁  ∂ℓ/∂f₁  ∂ℓ/∂h₂  ∂ℓ/∂f₂

  → Chaque ∂ℓ/∂fₖ donne ∂ℓ/∂βₖ et ∂ℓ/∂Ωₖ`,
        label: 'Fig. 7.3 — Forward (→) et Backward (←)',
      },
      {
        type: 'text',
        content: `## 7.2 — Règle de la chaîne\n\nConsidérons un toy example :\n\ny = cos(sin(x) + exp(x))\n\nLe forward décompose en étapes : f = sin(x), g = exp(x), h = f + g, y = cos(h).\nLe backward applique dy/dx = dy/dh · dh/df · df/dx + dy/dh · dh/dg · dg/dx.`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\ell}{\\partial \\mathbf{f}_k} = \\underbrace{\\frac{\\partial \\mathbf{h}_{k+1}}{\\partial \\mathbf{f}_k}}_{\\text{Jacobien ReLU}} \\cdot \\underbrace{\\frac{\\partial \\mathbf{f}_{k+1}}{\\partial \\mathbf{h}_{k+1}}}_{= \\boldsymbol{\\Omega}_{k+1}} \\cdot \\underbrace{\\frac{\\partial \\ell}{\\partial \\mathbf{f}_{k+1}}}_{\\text{récursion}}',
        label: 'Éq. 7.13 — Récursion backward (couche k)',
        highlightVar: 'grad',
      },
      {
        type: 'text',
        content: `## 7.3 — Gradients par rapport aux paramètres\n\nUne fois ∂ℓ/∂fₖ calculé (via la récursion backward), on obtient directement les gradients pour les poids et biais de la couche k :`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\frac{\\partial \\ell}{\\partial \\boldsymbol{\\beta}_k} &= \\frac{\\partial \\ell}{\\partial \\mathbf{f}_k} \\\\[6pt] \\frac{\\partial \\ell}{\\partial \\boldsymbol{\\Omega}_k} &= \\frac{\\partial \\ell}{\\partial \\mathbf{f}_k} \\cdot \\mathbf{h}_k^T \\end{aligned}',
        label: 'Éq. 7.24–7.25 — Gradients des paramètres',
        highlightVar: 'grad',
      },
      {
        type: 'text',
        content: `## 7.4 — Dérivée du ReLU\n\nLa dérivée de ReLU est triviale : elle crée une matrice diagonale avec 1 pour les entrées positives et 0 pour les négatives. C'est ce qui rend le backprop avec ReLU très rapide :`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\, \\text{ReLU}(z)}{\\partial z} = \\begin{cases} 0 & z < 0 \\\\ 1 & z > 0 \\end{cases} \\qquad \\Rightarrow \\qquad \\frac{\\partial \\mathbf{h}}{\\partial \\mathbf{f}} = \\text{diag}\\left[\\mathbb{I}[f_d > 0]\\right]',
        label: 'Éq. 7.15 — Jacobien du ReLU',
      },
      {
        type: 'text',
        content: `## 7.5 — Algorithme complet de backpropagation\n\n**Forward pass** : stocker toutes les pré-activations fₖ et activations hₖ\n**Backward pass** :\n1. Calculer ∂ℓ/∂fₖ pour K (dernière couche)\n2. Pour k = K−1, ..., 0 : propager ∂ℓ/∂fₖ\n3. À chaque couche : extraire ∂ℓ/∂βₖ et ∂ℓ/∂Ωₖ\n\n**Complexité** : O(paramètres) — identique au forward pass !`,
      },
      {
        type: 'callout',
        content: '⚡ **Autograd** de PyTorch implémente la **différentiation algorithmique** (reverse mode). Le graphe de calcul est construit automatiquement pendant le forward. Un seul appel à \\`loss.backward()\\` calcule TOUS les gradients. C\'est la version automatique de la backpropagation.',
      },
      {
        type: 'text',
        content: `## 7.6 — Initialisation de He\n\nSi les poids sont trop grands → les gradients **explosent**. Trop petits → ils **s'évanouissent**. L'initialisation de **He** (2015) choisit la variance des poids pour que les activations ReLU gardent une variance constante :`,
      },
      {
        type: 'equation',
        content: '\\sigma^2 = \\frac{2}{D_h} \\qquad \\Rightarrow \\qquad \\omega_{ij} \\sim \\mathcal{N}\\!\\left(0, \\frac{2}{D_h}\\right)',
        label: 'Éq. 7.40 — Initialisation de He (pour ReLU)',
        highlightVar: 'hidden',
      },
      {
        type: 'diagram',
        content: `  Sans bonne init                Avec He init
  ──────────────────            ──────────────────
  Couche 1:  σ = 1.0            Couche 1:  σ = 0.71
  Couche 5:  σ = 0.001          Couche 5:  σ = 0.68
  Couche 10: σ = 0.000001       Couche 10: σ = 0.65
  → Gradients ÉVANOUISSENT !    → Gradients STABLES ✓
  
  Formule : σ² = 2/Dₕ (facteur 2 pour ReLU qui clippe 50%)
  PyTorch : nn.init.kaiming_normal_(w, nonlinearity='relu')`,
        label: 'Fig. 7.8 — Variance des activations avec/sans He init',
      },
      {
        type: 'callout',
        content: '🧠 **Résumé Ch. 7** :\n(1) Forward = calculer et stocker les activations couche par couche\n(2) Backward = propager ∂ℓ/∂f de la sortie vers l\'entrée via la règle de la chaîne\n(3) Complexité backward = complexité forward (remarquable !)\n(4) PyTorch Autograd fait tout automatiquement\n(5) He init : σ² = 2/Dₕ pour stabiliser les gradients profonds',
      },
    ],
    exercises: [
      {
        id: 'bp-ex1',
        title: '💻 Pratique — Autograd en action',
        instructions: 'Utilisez PyTorch Autograd pour calculer les gradients d\'une expression y = w*x + b, loss = (y-10)². Vérifiez manuellement avec la règle de la chaîne.',
        starterCode: `import torch

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

# Vérification manuelle : ∂loss/∂w = 2(y-10) * ∂y/∂w = 2(y-10) * x
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
      {
        id: 'bp-th1',
        title: '🧠 Théorie — Backprop manuelle (toy example)',
        instructions: 'Implémentez le forward et backward pass manuellement (sans autograd) pour un réseau à 1 couche cachée. Vérifiez contre PyTorch.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(0)

# Réseau : x(1) → h(3) → y(1), ReLU, loss = MSE
D_i, D_h, D_o = 1, 3, 1

# Paramètres
W1 = torch.randn(D_h, D_i, requires_grad=True)  # (3,1)
b1 = torch.randn(D_h, requires_grad=True)        # (3,)
W2 = torch.randn(D_o, D_h, requires_grad=True)   # (1,3)
b2 = torch.randn(D_o, requires_grad=True)         # (1,)

x = torch.tensor([[1.5]])  # (1,1)
y_true = torch.tensor([[2.0]])

# ── Forward pass (manuel) ──
f0 = (W1 @ x.T).squeeze() + b1   # pré-activation
h1 = torch.relu(f0)               # activation
f1 = (W2 @ h1.unsqueeze(1)).squeeze() + b2  # sortie
loss = (f1 - y_true.squeeze()) ** 2

print(f"f0 = {f0.detach().tolist()}")
print(f"h1 = {h1.detach().tolist()}")
print(f"f1 = {f1.detach().item():.4f}")
print(f"loss = {loss.detach().item():.4f}")

# ── Backward pass (manuel) ──
dl_df1 = 2 * (f1 - y_true.squeeze())        # ∂L/∂f₁
dl_dW2 = ___  # dl_df1 * h1ᵀ
dl_db2 = ___  # dl_df1
dl_dh1 = ___  # W2ᵀ * dl_df1
dl_df0 = dl_dh1.squeeze() * (f0 > 0).float()  # ReLU mask
dl_dW1 = dl_df0.unsqueeze(1) @ x  # ∂L/∂W₁
dl_db1 = dl_df0                    # ∂L/∂b₁

# Vérifier avec autograd
loss.backward()
print(f"\\n{'Param':>6} │ {'Manuel':>10} │ {'Autograd':>10} │ {'Match':>5}")
print(f"{'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*5}")
for name, manual, auto in [('W2', dl_dW2, W2.grad),
                             ('b2', dl_db2, b2.grad),
                             ('W1', dl_dW1, W1.grad),
                             ('b1', dl_db1, b1.grad)]:
    m = manual.detach().flatten()
    a = auto.flatten()
    match = torch.allclose(m, a, atol=1e-5)
    print(f"{name:>6} │ {m.tolist()} │ {a.tolist()} │ {'✓' if match else '✗'}")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(0)

D_i, D_h, D_o = 1, 3, 1

W1 = torch.randn(D_h, D_i, requires_grad=True)
b1 = torch.randn(D_h, requires_grad=True)
W2 = torch.randn(D_o, D_h, requires_grad=True)
b2 = torch.randn(D_o, requires_grad=True)

x = torch.tensor([[1.5]])
y_true = torch.tensor([[2.0]])

f0 = (W1 @ x.T).squeeze() + b1
h1 = torch.relu(f0)
f1 = (W2 @ h1.unsqueeze(1)).squeeze() + b2
loss = (f1 - y_true.squeeze()) ** 2

print(f"f0 = {f0.detach().tolist()}")
print(f"h1 = {h1.detach().tolist()}")
print(f"f1 = {f1.detach().item():.4f}")
print(f"loss = {loss.detach().item():.4f}")

dl_df1 = 2 * (f1 - y_true.squeeze())
dl_dW2 = dl_df1 * h1.unsqueeze(0)
dl_db2 = dl_df1
dl_dh1 = W2.T * dl_df1
dl_df0 = dl_dh1.squeeze() * (f0 > 0).float()
dl_dW1 = dl_df0.unsqueeze(1) @ x
dl_db1 = dl_df0

loss.backward()
print(f"\\n{'Param':>6} │ {'Manuel':>10} │ {'Autograd':>10} │ {'Match':>5}")
print(f"{'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*5}")
for name, manual, auto in [('W2', dl_dW2, W2.grad),
                             ('b2', dl_db2, b2.grad),
                             ('W1', dl_dW1, W1.grad),
                             ('b1', dl_db1, b1.grad)]:
    m = manual.detach().flatten()
    a = auto.flatten()
    match = torch.allclose(m, a, atol=1e-5)
    print(f"{name:>6} │ {m.tolist()} │ {a.tolist()} │ {'✓' if match else '✗'}")`,
        hints: [
          '∂L/∂W₂ = ∂L/∂f₁ · h₁ᵀ (produit extérieur)',
          '∂L/∂h₁ = W₂ᵀ · ∂L/∂f₁',
          '∂L/∂f₀ = ∂L/∂h₁ · I[f₀>0] (masque ReLU)',
        ],
        completed: false,
      },
      {
        id: 'bp-pr2',
        title: '💻 Pratique — He init vs mauvaise init',
        instructions: 'Comparez l\'entraînement d\'un réseau profond (10 couches) avec initialisation standard vs He. Observez les gradients.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

def make_deep_net(init_type='default'):
    layers = []
    dims = [1] + [50]*10 + [1]
    for i in range(len(dims)-1):
        layer = nn.Linear(dims[i], dims[i+1])
        if init_type == 'he':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif init_type == 'small':
            nn.init.normal_(layer.weight, std=0.01)
        layers.append(layer)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

x = torch.randn(32, 1)
y = torch.sin(x)

for init in ['small', 'default', 'he']:
    model = make_deep_net(init)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    # Gradient de la PREMIÈRE couche
    grad_norm = model[0].weight.grad.norm().item()
    # Activation de la DERNIÈRE couche
    out_std = model(x).std().item()
    
    print(f"Init {init:>8s}: grad_norm_L0 = {grad_norm:.6f}, out_std = {out_std:.4f}")

print(f"\\n→ 'he' garde les gradients ni trop grands ni trop petits !")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

def make_deep_net(init_type='default'):
    layers = []
    dims = [1] + [50]*10 + [1]
    for i in range(len(dims)-1):
        layer = nn.Linear(dims[i], dims[i+1])
        if init_type == 'he':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif init_type == 'small':
            nn.init.normal_(layer.weight, std=0.01)
        layers.append(layer)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

x = torch.randn(32, 1)
y = torch.sin(x)

for init in ['small', 'default', 'he']:
    model = make_deep_net(init)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    grad_norm = model[0].weight.grad.norm().item()
    out_std = model(x).std().item()
    
    print(f"Init {init:>8s}: grad_norm_L0 = {grad_norm:.6f}, out_std = {out_std:.4f}")

print(f"\\n→ 'he' garde les gradients ni trop grands ni trop petits !")`,
        hints: [
          'nn.init.kaiming_normal_ implémente He init',
          'Avec std=0.01 les gradients seront quasi-nuls (vanishing)',
          'He init utilise σ² = 2/fan_in pour ReLU',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Backpropagation — Ch. 7 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Autograd simple ──
print("═══ Autograd ═══")
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b
loss = (y - 10) ** 2

loss.backward()
print(f"  y = {y.item():.2f}, loss = {loss.item():.2f}")
print(f"  ∂loss/∂w = {w.grad.item():.2f}")
print(f"  ∂loss/∂x = {x.grad.item():.2f}")
print(f"  ∂loss/∂b = {b.grad.item():.2f}")

# ── 2. Graphe de calcul complexe ──
print("\\n═══ Graphe complexe ═══")
a = torch.tensor(1.5, requires_grad=True)
b2 = torch.tensor(2.0, requires_grad=True)

c = a * b2
d = torch.relu(c - 2.5)
e = d ** 2

e.backward()
print(f"  e = {e.item():.4f}")
print(f"  ∂e/∂a = {a.grad.item():.4f}")
print(f"  ∂e/∂b = {b2.grad.item():.4f}")

# ── 3. He Initialization ──
print("\\n═══ He Init ═══")
D = 256
layer = nn.Linear(D, D)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
print(f"  E[w²] = {(layer.weight**2).mean().item():.6f}")
print(f"  2/D   = {2/D:.6f}")
print(f"  → {'✓ Match !' if abs((layer.weight**2).mean().item() - 2/D) < 0.01 else '✗'}")

# ── 4. Gradient flow dans un réseau profond ──
print("\\n═══ Gradient Flow (10 couches) ═══")
model = nn.Sequential(*[
    layer for D in [1] + [50]*10 + [1]
    for layer in [nn.Linear(D, 50), nn.ReLU()]
][:-1])
# He init
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

x = torch.randn(32, 1)
out = model(x)
loss = out.mean()
loss.backward()

for i, m in enumerate(model):
    if isinstance(m, nn.Linear):
        print(f"  Layer {i}: grad_norm = {m.weight.grad.norm().item():.6f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 8 — RÉGULARISATION (Ch. 9)
  // ═══════════════════════════════════════
  {
    id: 'regularization',
    title: 'Régularisation & Généralisation',
    shortTitle: 'Régular.',
    description: 'Biais-Variance, L2/Weight Decay, Dropout, BatchNorm, Early Stopping, Data Augmentation (Ch. 9 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['backprop'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `## 9.1 — Biais, Variance et Bruit\n\nL'erreur de généralisation se décompose en trois termes :\n\n- **Biais** : erreur due à un modèle trop simple (under-fitting)\n- **Variance** : sensibilité aux données d'entraînement (over-fitting)\n- **Bruit** : erreur irréductible dans les données\n\nUn modèle trop simple a un **biais élevé**. Un modèle trop complexe a une **variance élevée**. Le but est de trouver le juste milieu.`,
      },
      {
        type: 'diagram',
        content: `  Erreur ↑
    │\\
    │ \\  Biais
    │  \\          ╱ Variance
    │   \\       ╱
    │    \\    ╱
    │     \\╱──── Erreur totale
    │     ╱\\
    │   ╱   \\
    └──────────────── Complexité du modèle →
      Simple              Complexe
    (underfitting)      (overfitting)
                   ↑
            Sweet spot !`,
        label: 'Fig. 9.2 — Compromis Biais-Variance',
      },
      {
        type: 'text',
        content: `## 9.2 — Régularisation L2 (Weight Decay)\n\nOn ajoute un terme qui pénalise les **poids trop grands**. Cela force le modèle à utiliser des poids plus petits → solutions plus lisses → meilleure généralisation. En pratique, c'est le paramètre \`weight_decay\` de l'optimiseur :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{\\text{reg}} = \\underbrace{\\mathcal{L}_{\\text{data}}}_{\\text{MSE ou CE}} + \\underbrace{\\frac{\\lambda}{2} \\| \\boldsymbol{\\phi} \\|_2^2}_{\\text{pénalité L2}}',
        label: 'Éq. 9.3 — Régularisation L2',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `## 9.3 — Dropout\n\nPendant l'entraînement, on **désactive aléatoirement** une fraction p des neurones. Cela force le réseau à ne pas dépendre d'un neurone unique → équivalent approximatif d'un **ensemble** de sous-réseaux.\n\nÀ l'inférence (\`model.eval()\`), tous les neurones sont actifs mais multipliés par (1−p) pour compenser.`,
      },
      {
        type: 'equation',
        content: '\\tilde{h}_d = h_d \\cdot m_d \\quad \\text{où } m_d \\sim \\text{Bernoulli}(1-p)',
        label: 'Éq. 9.14 — Dropout mask',
      },
      {
        type: 'text',
        content: `## 9.4 — Batch Normalization\n\nNormalise les activations de chaque couche pour avoir **μ=0, σ=1** sur le mini-batch, puis applique un rescaling appris γ,β. Agit comme régularisateur ET accélérateur :`,
      },
      {
        type: 'equation',
        content: '\\hat{h}_d = \\gamma_d \\cdot \\frac{h_d - \\mu_{\\mathcal{B}}}{\\sqrt{\\sigma_{\\mathcal{B}}^2 + \\epsilon}} + \\beta_d',
        label: 'Éq. — Batch Normalization',
      },
      {
        type: 'text',
        content: `## 9.5 — Autres techniques\n\n- **Early Stopping** : surveiller la loss de validation, arrêter quand elle augmente\n- **Data Augmentation** : créer des exemples artificiels (rotations, flips, crops, color jitter)\n- **Label Smoothing** : remplacer les labels one-hot par des soft labels (1 → 0.9, 0 → 0.1/K)\n- **Weight Noise / Gradient Noise** : ajouter du bruit pendant l'entraînement`,
      },
      {
        type: 'callout',
        content: '💡 **Recette anti-overfitting** :\n1. Plus de **données** (data augmentation)\n2. **Weight Decay** (λ = 1e-4 ou 1e-5)\n3. **Dropout** (p = 0.1 à 0.5)\n4. **Early Stopping** (patience 5-10 epochs)\n5. **Batch Norm** (accélère + régularise)\n6. Réduire la **taille du modèle** (dernier recours)',
      },
    ],
    exercises: [
      {
        id: 'reg-ex1',
        title: '💻 Pratique — Dropout train vs eval',
        instructions: 'Montrez la différence entre model.train() et model.eval() avec Dropout. Vérifiez que les sorties sont stochastiques en train mode et déterministes en eval mode.',
        starterCode: `import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(50, 50), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(50, 1)
)

x = torch.randn(1, 10)

# Train mode → dropout actif → sorties différentes
model.train()
out1 = model(x).item()
out2 = model(x).item()
print(f"Train mode: {out1:.4f} vs {out2:.4f} → {'Différents ✓' if abs(out1-out2) > 1e-6 else 'ERREUR'}")

# Eval mode → dropout désactivé → sorties identiques
model.___()  # passer en eval
out3 = model(x).item()
out4 = model(x).item()
print(f"Eval mode:  {out3:.4f} vs {out4:.4f} → {'Identiques ✓' if abs(out3-out4) < 1e-6 else 'ERREUR'}")`,
        solution: `import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(50, 50), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(50, 1)
)

x = torch.randn(1, 10)

model.train()
out1 = model(x).item()
out2 = model(x).item()
print(f"Train mode: {out1:.4f} vs {out2:.4f} → {'Différents ✓' if abs(out1-out2) > 1e-6 else 'ERREUR'}")

model.eval()
out3 = model(x).item()
out4 = model(x).item()
print(f"Eval mode:  {out3:.4f} vs {out4:.4f} → {'Identiques ✓' if abs(out3-out4) < 1e-6 else 'ERREUR'}")`,
        hints: [
          'model.eval() désactive le dropout et la batch norm',
          'model.train() les réactive',
        ],
        completed: false,
      },
      {
        id: 'reg-ex2',
        title: '💻 Pratique — Overfitting vs Régularisation',
        instructions: 'Entraînez un modèle sur peu de données avec et sans régularisation. Observez l\'overfitting et l\'effet de weight decay + dropout.',
        starterCode: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Peu de données (overfitting garanti !)
x_train = torch.randn(20, 1)
y_train = torch.sin(x_train) + 0.1 * torch.randn(20, 1)
x_test = torch.linspace(-4, 4, 100).unsqueeze(1)
y_test = torch.sin(x_test)

# Modèle SANS régularisation
model_noreg = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(),
    nn.Linear(100, 100), nn.ReLU(),
    nn.Linear(100, 1)
)

# Modèle AVEC régularisation
model_reg = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(100, 1)
)

opt_noreg = optim.Adam(model_noreg.parameters(), lr=0.01)
opt_reg = optim.Adam(model_reg.parameters(), lr=0.01, weight_decay=___)

loss_fn = nn.MSELoss()

for epoch in range(500):
    for model, opt in [(model_noreg, opt_noreg), (model_reg, opt_reg)]:
        model.train()
        loss = loss_fn(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

# Évaluation
model_noreg.eval(); model_reg.eval()
test_loss_noreg = loss_fn(model_noreg(x_test), y_test).item()
test_loss_reg = loss_fn(model_reg(x_test), y_test).item()
train_loss_noreg = loss_fn(model_noreg(x_train), y_train).item()
train_loss_reg = loss_fn(model_reg(x_train), y_train).item()

print(f"{'':>10} │ {'Train':>8} │ {'Test':>8} │ {'Gap':>8}")
print(f"{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
print(f"{'No reg':>10} │ {train_loss_noreg:8.4f} │ {test_loss_noreg:8.4f} │ {test_loss_noreg-train_loss_noreg:8.4f}")
print(f"{'Dropout+WD':>10} │ {train_loss_reg:8.4f} │ {test_loss_reg:8.4f} │ {test_loss_reg-train_loss_reg:8.4f}")`,
        solution: `import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x_train = torch.randn(20, 1)
y_train = torch.sin(x_train) + 0.1 * torch.randn(20, 1)
x_test = torch.linspace(-4, 4, 100).unsqueeze(1)
y_test = torch.sin(x_test)

model_noreg = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(),
    nn.Linear(100, 100), nn.ReLU(),
    nn.Linear(100, 1)
)

model_reg = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(100, 1)
)

opt_noreg = optim.Adam(model_noreg.parameters(), lr=0.01)
opt_reg = optim.Adam(model_reg.parameters(), lr=0.01, weight_decay=1e-4)

loss_fn = nn.MSELoss()

for epoch in range(500):
    for model, opt in [(model_noreg, opt_noreg), (model_reg, opt_reg)]:
        model.train()
        loss = loss_fn(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

model_noreg.eval(); model_reg.eval()
test_loss_noreg = loss_fn(model_noreg(x_test), y_test).item()
test_loss_reg = loss_fn(model_reg(x_test), y_test).item()
train_loss_noreg = loss_fn(model_noreg(x_train), y_train).item()
train_loss_reg = loss_fn(model_reg(x_train), y_train).item()

print(f"{'':>10} │ {'Train':>8} │ {'Test':>8} │ {'Gap':>8}")
print(f"{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
print(f"{'No reg':>10} │ {train_loss_noreg:8.4f} │ {test_loss_noreg:8.4f} │ {test_loss_noreg-train_loss_noreg:8.4f}")
print(f"{'Dropout+WD':>10} │ {train_loss_reg:8.4f} │ {test_loss_reg:8.4f} │ {test_loss_reg-train_loss_reg:8.4f}")`,
        hints: [
          'weight_decay=1e-4 est une bonne valeur par défaut',
          'Le "gap" train-test mesure l\'overfitting',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Régularisation — Ch. 9 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Weight Decay (L2) ──
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
print("✓ Weight decay λ=0.01")

# ── 2. Dropout ──
model_drop = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(50, 50), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(50, 1)
)

x = torch.randn(1, 10)
model_drop.train()
print(f"\\nTrain: {model_drop(x).item():.4f} vs {model_drop(x).item():.4f} (différents)")
model_drop.eval()
print(f"Eval:  {model_drop(x).item():.4f} vs {model_drop(x).item():.4f} (identiques)")

# ── 3. Batch Normalization ──
bn_model = nn.Sequential(
    nn.Linear(10, 50), nn.BatchNorm1d(50), nn.ReLU(),
    nn.Linear(50, 1)
)
print(f"\\n✓ BatchNorm model: {sum(p.numel() for p in bn_model.parameters())} params")

# ── 4. Ensemble des techniques ──
full_model = nn.Sequential(
    nn.Linear(10, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 1)
)
print(f"✓ Full model: {sum(p.numel() for p in full_model.parameters())} params")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 9 — CNN (Ch. 10)
  // ═══════════════════════════════════════
  {
    id: 'cnn',
    title: 'Réseaux Convolutifs (CNN)',
    shortTitle: 'CNN',
    description: 'Convolutions 2D, invariance/équivariance, pooling, architectures classiques (Ch. 10 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['regularization'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `## 10.1 — Invariance et Équivariance\n\nLes images ont une structure spatiale : les mêmes patterns (bords, textures) apparaissent partout. Deux concepts formalisent cette idée :\n\n- **Invariance** : f[t[x]] = f[x] — la sortie ne change pas (classification : "c'est un chat" peu importe la position)\n- **Équivariance** : f[t[x]] = t[f[x]] — la sortie se transforme identiquement (détection d'objets : bouger l'objet → bouger la boîte)`,
      },
      {
        type: 'text',
        content: `## 10.2 — Convolution 1D et 2D\n\nLa **convolution** applique un filtre (kernel) qui **glisse** sur l'entrée. Les mêmes poids sont partagés partout → **équivariance à la translation** et réduction massive du nombre de paramètres. La convolution 2D est le cœur des CNN :`,
      },
      {
        type: 'equation',
        content: 'z_{ij} = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1} \\omega_{mn} \\cdot x_{i+m, \\, j+n} + b',
        label: 'Éq. 10.3 — Convolution 2D (kernel M×N)',
      },
      {
        type: 'diagram',
        content: `  Input (5×5)          Kernel (3×3)       Output (3×3)
  ┌─┬─┬─┬─┬─┐         ┌─┬─┬─┐
  │·│·│·│ │ │         │1│0│1│           ┌─┬─┬─┐
  ├─┼─┼─┼─┼─┤    ∗    ├─┼─┼─┤    =     │ │ │ │
  │·│·│·│ │ │         │0│1│0│           ├─┼─┼─┤
  ├─┼─┼─┼─┼─┤         ├─┼─┼─┤           │ │ │ │
  │·│·│·│ │ │         │1│0│1│           ├─┼─┼─┤
  ├─┼─┼─┼─┼─┤         └─┴─┴─┘           │ │ │ │
  │ │ │ │ │ │    ← 9 poids partagés     └─┴─┴─┘
  ├─┼─┼─┼─┼─┤      partout !
  │ │ │ │ │ │
  └─┴─┴─┴─┴─┘    padding='same' → taille conservée`,
        label: 'Fig. 10.5 — Convolution 2D sliding window',
      },
      {
        type: 'text',
        content: `## 10.3 — Canaux, Stride, Padding\n\n- **Canaux d'entrée Cᵢₙ** : image RGB → 3 canaux. Le kernel est 3D : (Cᵢₙ × K × K)\n- **Canaux de sortie Cₒᵤₜ** : nombre de filtres → feature maps. Paramètres = Cₒᵤₜ × Cᵢₙ × K × K\n- **Stride** : pas du glissement (stride=2 → divise la taille par 2)\n- **Padding** : ajout de zéros autour pour contrôler la taille de sortie`,
      },
      {
        type: 'equation',
        content: 'H_{\\text{out}} = \\left\\lfloor \\frac{H_{\\text{in}} + 2p - k}{s} \\right\\rfloor + 1',
        label: 'Taille de sortie d\'une convolution',
      },
      {
        type: 'text',
        content: `## 10.4 — Pooling\n\n**Max Pooling** : prend le maximum dans chaque fenêtre → réduit la résolution + ajoute une petite **invariance à la translation**.\n**Average Pooling** : prend la moyenne.\n**Global Average Pooling** : une seule valeur par channel → remplace le Flatten+FC final.`,
      },
      {
        type: 'text',
        content: `## 10.5 — Architecture CNN typique\n\nEmpiler : [Conv → BatchNorm → ReLU → Pool]×N → Flatten → FC → Sortie\n\n- Les premières couches détectent des **features bas-niveau** (bords, coins)\n- Les couches profondes combinent en **features haut-niveau** (textures, objets)\n- En augmentant les canaux et réduisant la résolution spatiale :`,
      },
      {
        type: 'diagram',
        content: `  Input    Conv1+Pool  Conv2+Pool  Conv3+Pool  FC
  ────────────────────────────────────────────────
  28×28×1  → 14×14×16  → 7×7×32   → 3×3×64   → 10
  
  Résolution: ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
  Channels:   ↑↑↑↑↑↑↑↑↑↑↑↑↑↑
  
  Hiérarchie : bords → textures → parties → objets`,
        label: 'Fig. — Progression spatiale dans un CNN',
      },
      {
        type: 'callout',
        content: '⚡ **Architectures célèbres** :\n• **LeNet-5** (1998) : 2 conv, 60K params → MNIST\n• **AlexNet** (2012) : 5 conv, 60M params → ImageNet revolution\n• **VGG** (2014) : 16-19 couches, pattern 3×3 → simple et profond\n• **GoogLeNet** (2014) : Inception modules → parallélisme\n• **ResNet** (2015) : Skip connections → 152+ couches',
      },
    ],
    exercises: [
      {
        id: 'cnn-ex1',
        title: '💻 Pratique — CNN pour MNIST',
        instructions: 'Créez un CNN avec 2 couches conv pour classifier des images 28×28 (1 canal). Calculez les dimensions de chaque couche.',
        starterCode: `import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 1→16 channels, kernel 3, padding 1 → 28×28×16
        # MaxPool: → 14×14×16
        self.conv1 = ___
        # Conv2: 16→32 channels, kernel 3, padding 1 → 14×14×32
        # MaxPool: → 7×7×32
        self.conv2 = ___
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = ___   # 32*7*7 → 10
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28→14
        x = self.pool(self.relu(self.conv2(x)))  # 14→7
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

model = SimpleCNN()
x = torch.randn(4, 1, 28, 28)
out = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`,
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
x = torch.randn(4, 1, 28, 28)
out = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`,
        hints: [
          'nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)',
          'Après 2 MaxPool(2,2) sur 28×28 → 7×7',
          'nn.Linear(32 * 7 * 7, 10)',
        ],
        completed: false,
      },
      {
        id: 'cnn-th1',
        title: '🧠 Théorie — Comptage des paramètres conv',
        instructions: 'Calculez le nombre de paramètres de chaque couche conv et comparez avec un réseau fully-connected équivalent.',
        starterCode: `import torch
import torch.nn as nn

# CNN
conv1 = nn.Conv2d(3, 16, 3, padding=1)   # RGB input
conv2 = nn.Conv2d(16, 32, 3, padding=1)
conv3 = nn.Conv2d(32, 64, 3, padding=1)

# Fully-connected équivalent (32×32 RGB image → même features)
fc1 = nn.Linear(3 * 32 * 32, 16 * 32 * 32)  # même "capacité"

print("═══ CNN vs FC — Nombre de paramètres ═══")
print(f"Conv1 (3→16, 3×3):  {sum(p.numel() for p in conv1.parameters()):>10,}")
print(f"Conv2 (16→32, 3×3): {sum(p.numel() for p in conv2.parameters()):>10,}")
print(f"Conv3 (32→64, 3×3): {sum(p.numel() for p in conv3.parameters()):>10,}")
total_conv = sum(sum(p.numel() for p in m.parameters()) for m in [conv1, conv2, conv3])
print(f"Total CNN:           {total_conv:>10,}")
print(f"\\nFC1 (3*32*32 → 16*32*32): {sum(p.numel() for p in fc1.parameters()):>10,}")
print(f"\\n→ Le FC a {sum(p.numel() for p in fc1.parameters()) // total_conv}× plus de paramètres !")
print(f"   grâce au partage de poids (weight sharing) des convolutions.")`,
        solution: `import torch
import torch.nn as nn

conv1 = nn.Conv2d(3, 16, 3, padding=1)
conv2 = nn.Conv2d(16, 32, 3, padding=1)
conv3 = nn.Conv2d(32, 64, 3, padding=1)

fc1 = nn.Linear(3 * 32 * 32, 16 * 32 * 32)

print("═══ CNN vs FC — Nombre de paramètres ═══")
print(f"Conv1 (3→16, 3×3):  {sum(p.numel() for p in conv1.parameters()):>10,}")
print(f"Conv2 (16→32, 3×3): {sum(p.numel() for p in conv2.parameters()):>10,}")
print(f"Conv3 (32→64, 3×3): {sum(p.numel() for p in conv3.parameters()):>10,}")
total_conv = sum(sum(p.numel() for p in m.parameters()) for m in [conv1, conv2, conv3])
print(f"Total CNN:           {total_conv:>10,}")
print(f"\\nFC1 (3*32*32 → 16*32*32): {sum(p.numel() for p in fc1.parameters()):>10,}")
print(f"\\n→ Le FC a {sum(p.numel() for p in fc1.parameters()) // total_conv}× plus de paramètres !")
print(f"   grâce au partage de poids (weight sharing) des convolutions.")`,
        hints: [
          'Conv2d params = Cout × (Cin × K × K + 1_bias)',
          'Le weight sharing réduit drastiquement les paramètres',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Réseaux Convolutifs — Ch. 10 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Convolution 2D ──
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(1, 1, 28, 28)
out = conv(x)
print(f"Conv2d: {x.shape} → {out.shape}")
print(f"Kernel: {conv.weight.shape} = {conv.weight.numel()} poids + {conv.bias.numel()} biais")

# ── 2. CNN complet ──
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

model = MNISTNet()
out = model(torch.randn(4, 1, 28, 28))
print(f"\\nMNISTNet: batch=4 → {out.shape}")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

# ── 3. Feature maps ──
print("\\n═══ Feature maps par couche ═══")
x = torch.randn(1, 1, 28, 28)
for name, layer in model.features.named_children():
    x = layer(x)
    print(f"  Layer {name}: {x.shape}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 10 — RESIDUAL NETWORKS (Ch. 11)
  // ═══════════════════════════════════════
  {
    id: 'resnet',
    title: 'Réseaux Résiduels (ResNet)',
    shortTitle: 'ResNet',
    description: 'Skip connections, blocs résiduels, BatchNorm, entraîner des réseaux de 100+ couches (Ch. 11 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['cnn'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `## 11.1 — Le problème de la profondeur\n\nLes réseaux très profonds (>20 couches) souffrent du **degradation problem** : la performance se dégrade avec la profondeur, même sur le jeu d'entraînement ! Ce n'est pas de l'overfitting — c'est un problème d'**optimisation** dû aux gradients évanescents.`,
      },
      {
        type: 'text',
        content: `## 11.2 — Connexion résiduelle\n\nL'idée géniale de He et al. (2015) : au lieu d'apprendre la transformation complète h → h', le réseau apprend le **résidu** f(h) = h' − h. La sortie est simplement h + f(h). Si le résidu est proche de zéro, le gradient circule librement via le "highway" de la skip connection :`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}_{k+1} = \\mathbf{h}_k + f_k(\\mathbf{h}_k) \\qquad \\text{(résiduel)}',
        label: 'Éq. 11.1 — Skip connection',
        highlightVar: 'hidden',
      },
      {
        type: 'diagram',
        content: `  ┌──────────────────────────────────────┐
  │         Skip Connection              │
  │              ╭───────────────╮       │
  │    x  ──────▶│               │──▶ +  │──▶ output
  │    │         │  Conv-BN-ReLU │       ▲
  │    │         │  Conv-BN      │       │
  │    │         ╰───────────────╯       │
  │    ╰─────────────────────────────────╯
  │         x directement additionné !
  └──────────────────────────────────────┘
  
  Si f(x) → 0, alors output ≈ x (identité)
  → Le réseau peut toujours "copier" l'entrée`,
        label: 'Fig. 11.1 — Bloc résiduel avec skip connection',
      },
      {
        type: 'text',
        content: `## 11.3 — Bloc résiduel pré-activation\n\nDeux variantes :\n- **Post-activation** (ResNet v1) : Conv → BN → ReLU → Conv → BN → + → ReLU\n- **Pré-activation** (ResNet v2, meilleur) : BN → ReLU → Conv → BN → ReLU → Conv → +\n\nLa version pré-activation garde le chemin résiduel **propre** (pas de non-linéarité sur le shortcut).`,
      },
      {
        type: 'text',
        content: `## 11.4 — Changement de dimensions\n\nQuand les dimensions changent (doubler les channels, réduire la résolution), la skip connection utilise une **convolution 1×1** avec stride 2 pour adapter les dimensions. Cela garde l'addition h + f(h) valide.`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\ell}{\\partial \\mathbf{h}_k} = \\frac{\\partial \\ell}{\\partial \\mathbf{h}_{k+1}} \\cdot \\left( \\mathbf{I} + \\frac{\\partial f_k}{\\partial \\mathbf{h}_k} \\right)',
        label: 'Gradient : le terme I empêche le vanishing !',
        highlightVar: 'grad',
      },
      {
        type: 'callout',
        content: '🧠 **Pourquoi ça marche ?** Le gradient de la skip connection contient un terme **identité I**. Même si ∂f/∂h est petit, le gradient circule via I. Cela crée un "highway" pour le gradient, permettant d\'entraîner des réseaux de **152, 1000+** couches !',
      },
      {
        type: 'text',
        content: `## 11.5 — Architectures ResNet\n\n- **ResNet-18/34** : blocs basiques (2 convolutions 3×3)\n- **ResNet-50/101/152** : blocs bottleneck (1×1 → 3×3 → 1×1, réduit les calculs)\n- **WideResNet** : plus large au lieu de plus profond\n- **ResNeXt** : blocs parallèles avec cardinality\n- **DenseNet** : chaque couche connectée à TOUTES les précédentes`,
      },
    ],
    exercises: [
      {
        id: 'resnet-ex1',
        title: '💻 Pratique — Bloc résiduel',
        instructions: 'Implémentez un bloc résiduel (Residual Block) et un SimpleResNet avec 3 blocs. Vérifiez que les gradients circulent bien.',
        starterCode: `import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
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
        return ___  # skip connection !

class SimpleResNet(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleResNet(num_blocks=5)
x = torch.randn(2, 1, 28, 28)
out = model(x)
loss = out.sum()
loss.backward()

print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Conv1 grad norm: {model.conv1.weight.grad.norm().item():.6f}")
print(f"→ Le gradient circule bien malgré 11 couches conv !")`,
        solution: `import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
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
        return x + self.block(x)

class SimpleResNet(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleResNet(num_blocks=5)
x = torch.randn(2, 1, 28, 28)
out = model(x)
loss = out.sum()
loss.backward()

print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Conv1 grad norm: {model.conv1.weight.grad.norm().item():.6f}")
print(f"→ Le gradient circule bien malgré 11 couches conv !")`,
        hints: [
          'return x + self.block(x)  — c\'est la skip connection',
          'Le + additionne l\'entrée et la sortie du bloc',
        ],
        completed: false,
      },
      {
        id: 'resnet-th1',
        title: '🧠 Théorie — ResNet vs PlainNet (gradient flow)',
        instructions: 'Comparez le flux de gradient dans un réseau "plain" (sans skip) vs ResNet de même profondeur. Montrez que ResNet préserve les gradients.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

def make_plain_net(depth=20, ch=32):
    layers = [nn.Conv2d(1, ch, 3, padding=1)]
    for _ in range(depth):
        layers.extend([nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()])
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
    def forward(self, x):
        return x + self.net(x)

def make_resnet(depth=20, ch=32):
    layers = [nn.Conv2d(1, ch, 3, padding=1)]
    for _ in range(depth):
        layers.append(ResBlock(ch))
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)

x = torch.randn(1, 1, 28, 28)

for name, model_fn in [('Plain', make_plain_net), ('ResNet', make_resnet)]:
    model = model_fn(depth=20)
    out = model(x)
    out.sum().backward()
    grad = model[0].weight.grad.norm().item()
    print(f"{name:>6}: grad_norm_layer0 = {grad:.8f}")

print(f"\\n→ Le gradient du PlainNet est beaucoup plus petit (vanishing) !")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

def make_plain_net(depth=20, ch=32):
    layers = [nn.Conv2d(1, ch, 3, padding=1)]
    for _ in range(depth):
        layers.extend([nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()])
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
    def forward(self, x):
        return x + self.net(x)

def make_resnet(depth=20, ch=32):
    layers = [nn.Conv2d(1, ch, 3, padding=1)]
    for _ in range(depth):
        layers.append(ResBlock(ch))
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)

x = torch.randn(1, 1, 28, 28)

for name, model_fn in [('Plain', make_plain_net), ('ResNet', make_resnet)]:
    model = model_fn(depth=20)
    out = model(x)
    out.sum().backward()
    grad = model[0].weight.grad.norm().item()
    print(f"{name:>6}: grad_norm_layer0 = {grad:.8f}")

print(f"\\n→ Le gradient du PlainNet est beaucoup plus petit (vanishing) !")`,
        hints: [
          'Le PlainNet perd le gradient car chaque couche le multiplie par < 1',
          'Le ResNet préserve le gradient grâce au terme identité I',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Réseaux Résiduels — Ch. 11 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    
    def forward(self, x):
        return x + self.block(x)

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
print(f"Profondeur: 1 + 6 = 7 couches conv (3 blocs × 2)")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 11 — RNN/LSTM (Ch. 12)
  // ═══════════════════════════════════════
  {
    id: 'rnn',
    title: 'Réseaux Récurrents (RNN/LSTM)',
    shortTitle: 'RNN',
    description: 'Traitement séquentiel, état caché, vanishing gradient temporel, portes LSTM/GRU (Ch. 12 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['regularization'],
    category: 'architectures',
    theory: [
      {
        type: 'text',
        content: `## 12.1 — Pourquoi les séquences sont spéciales\n\nLes données séquentielles (texte, audio, séries temporelles) ont des **dépendances temporelles** : chaque élément dépend des précédents. Un réseau feedforward traite chaque entrée indépendamment — il ignore l'ordre. Le **RNN** résout cela avec un **état caché** qui accumule l'information au fil du temps.`,
      },
      {
        type: 'text',
        content: `## 12.2 — RNN (Recurrent Neural Network)\n\nÀ chaque pas de temps t, le RNN reçoit l'entrée xₜ et l'état précédent hₜ₋₁, et produit un nouvel état hₜ. Les mêmes poids W sont **partagés** à chaque pas de temps :`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}_t = \\tanh\\!\\left(\\mathbf{W}_{hh} \\mathbf{h}_{t-1} + \\mathbf{W}_{xh} \\mathbf{x}_t + \\mathbf{b}_h\\right)',
        label: 'Éq. 12.1 — RNN : État caché',
      },
      {
        type: 'diagram',
        content: `  Unfolded RNN (3 pas de temps)
  ─────────────────────────────────────────────
      x₁         x₂         x₃
      │          │          │
      ▼          ▼          ▼
  ┌───────┐ ┌───────┐ ┌───────┐
  │ RNN   │→│ RNN   │→│ RNN   │→ hₜ
  │ cell  │ │ cell  │ │ cell  │
  └───────┘ └───────┘ └───────┘
  h₀    h₁       h₂       h₃
  
  MÊMES POIDS (Whh, Wxh, bh) partagés à chaque t
  → "Dérouler" le RNN = réseau profond de T couches`,
        label: 'Fig. 12.2 — RNN déroulé dans le temps',
      },
      {
        type: 'text',
        content: `## 12.3 — Vanishing gradient temporel\n\nQuand on déroule le RNN sur T pas de temps, le backprop traverse T copies de W. Si les valeurs propres de W < 1, le gradient **s'évanouit** exponentiellement. Si > 1, il **explose**. C'est le problème des **long-range dependencies**.`,
      },
      {
        type: 'text',
        content: `## 12.4 — LSTM (Long Short-Term Memory)\n\nLe LSTM résout le vanishing gradient avec une **mémoire à long terme** (cell state cₜ) protégée par 3 **portes** (gates) apprenables :`,
      },
      {
        type: 'equation',
        content: '\\begin{aligned} \\mathbf{f}_t &= \\sigma(\\mathbf{W}_f [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_f) & \\text{(forget gate)} \\\\ \\mathbf{i}_t &= \\sigma(\\mathbf{W}_i [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_i) & \\text{(input gate)} \\\\ \\tilde{\\mathbf{c}}_t &= \\tanh(\\mathbf{W}_c [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_c) & \\text{(candidate)} \\\\ \\mathbf{c}_t &= \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t & \\text{(cell update)} \\\\ \\mathbf{o}_t &= \\sigma(\\mathbf{W}_o [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_o) & \\text{(output gate)} \\\\ \\mathbf{h}_t &= \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t) & \\text{(hidden state)} \\end{aligned}',
        label: 'Éq. 12.8–12.13 — LSTM complet',
      },
      {
        type: 'diagram',
        content: `  LSTM Cell
  ╔═══════════════════════════════════════╗
  ║  cₜ₋₁ ──▶ ×(fₜ) ──▶ + ──▶ cₜ        ║
  ║                     ▲                ║
  ║                 ×(iₜ)                ║
  ║                     ▲                ║
  ║                 tanh(c̃ₜ)             ║
  ║                                      ║
  ║  hₜ₋₁ ─┬──▶ [fₜ, iₜ, c̃ₜ, oₜ]       ║
  ║  xₜ   ─┘                            ║
  ║                                      ║
  ║  hₜ = oₜ × tanh(cₜ)                  ║
  ╚═══════════════════════════════════════╝
  
  fₜ : forget gate  → quoi effacer de cₜ₋₁
  iₜ : input gate   → quoi ajouter
  oₜ : output gate  → quoi exposer`,
        label: 'Fig. 12.6 — Architecture LSTM',
      },
      {
        type: 'text',
        content: `## 12.5 — GRU (Gated Recurrent Unit)\n\nVersion simplifiée du LSTM avec seulement **2 portes** (reset r et update z). Moins de paramètres, performances souvent comparables. Pas de cell state séparé.`,
      },
      {
        type: 'callout',
        content: '💡 **En pratique** :\n• \\`nn.RNN\\` : simple mais vanishing gradient → rarement utilisé\n• \\`nn.LSTM\\` : robuste, gère les longues séquences\n• \\`nn.GRU\\` : léger, bon pour les petits datasets\n• Les **Transformers** ont largement remplacé les RNN pour le NLP, mais les RNN restent utiles pour le streaming et les séquences très longues.',
      },
    ],
    exercises: [
      {
        id: 'rnn-ex1',
        title: '💻 Pratique — LSTM pour séquences',
        instructions: 'Construisez un classifieur de séquences avec LSTM. Utilisez le dernier état caché pour la classification.',
        starterCode: `import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim)
        last_hidden = ___  # dernier layer, dernier état
        return self.fc(last_hidden)

model = LSTMClassifier(input_dim=10, hidden_dim=64, num_classes=5)
x = torch.randn(8, 20, 10)  # batch=8, seq_len=20, features=10
out = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`,
        solution: `import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

model = LSTMClassifier(input_dim=10, hidden_dim=64, num_classes=5)
x = torch.randn(8, 20, 10)
out = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`,
        hints: [
          'h_n[-1] donne le dernier layer du dernier timestep',
          'h_n shape = (num_layers, batch, hidden_dim)',
        ],
        completed: false,
      },
      {
        id: 'rnn-th1',
        title: '🧠 Théorie — RNN vs LSTM gradient flow',
        instructions: 'Comparez la norme des gradients sur des séquences de longueur croissante pour un RNN simple vs LSTM.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

def test_gradient_flow(model_class, seq_lengths, input_dim=5, hidden_dim=32):
    results = []
    for T in seq_lengths:
        model = model_class(input_dim, hidden_dim, batch_first=True)
        x = torch.randn(1, T, input_dim, requires_grad=True)
        
        output, _ = model(x)
        loss = output[:, -1, :].sum()  # utiliser le dernier output
        loss.backward()
        
        grad_norm = x.grad[:, 0, :].norm().item()  # gradient au PREMIER timestep
        results.append(grad_norm)
    return results

seq_lengths = [5, 10, 20, 50, 100]
rnn_grads = test_gradient_flow(nn.RNN, seq_lengths)
lstm_grads = test_gradient_flow(nn.LSTM, seq_lengths)

print(f"{'T':>5} │ {'RNN grad':>12} │ {'LSTM grad':>12}")
print(f"{'─'*5}─┼─{'─'*12}─┼─{'─'*12}")
for T, rg, lg in zip(seq_lengths, rnn_grads, lstm_grads):
    print(f"{T:5d} │ {rg:12.6f} │ {lg:12.6f}")
print(f"\\n→ Le gradient RNN s'évanouit, le LSTM le préserve !")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

def test_gradient_flow(model_class, seq_lengths, input_dim=5, hidden_dim=32):
    results = []
    for T in seq_lengths:
        model = model_class(input_dim, hidden_dim, batch_first=True)
        x = torch.randn(1, T, input_dim, requires_grad=True)
        
        output, _ = model(x)
        loss = output[:, -1, :].sum()
        loss.backward()
        
        grad_norm = x.grad[:, 0, :].norm().item()
        results.append(grad_norm)
    return results

seq_lengths = [5, 10, 20, 50, 100]
rnn_grads = test_gradient_flow(nn.RNN, seq_lengths)
lstm_grads = test_gradient_flow(nn.LSTM, seq_lengths)

print(f"{'T':>5} │ {'RNN grad':>12} │ {'LSTM grad':>12}")
print(f"{'─'*5}─┼─{'─'*12}─┼─{'─'*12}")
for T, rg, lg in zip(seq_lengths, rnn_grads, lstm_grads):
    print(f"{T:5d} │ {rg:12.6f} │ {lg:12.6f}")
print(f"\\n→ Le gradient RNN s'évanouit, le LSTM le préserve !")`,
        hints: [
          'Le gradient au premier timestep mesure les long-range dependencies',
          'Le cell state du LSTM crée un "highway" pour le gradient',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Réseaux Récurrents — Ch. 12 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. RNN simple ──
print("═══ RNN ═══")
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(1, 5, 10)
output, h_n = rnn(x)
print(f"Output: {output.shape}  (batch, seq_len, hidden)")
print(f"Hidden: {h_n.shape}    (layers, batch, hidden)")

# ── 2. LSTM ──
print("\\n═══ LSTM ═══")
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)
print(f"Output: {output.shape}")
print(f"Hidden: {h_n.shape}, Cell: {c_n.shape}")

# ── 3. GRU ──
print("\\n═══ GRU ═══")
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, h_n = gru(x)
print(f"Output: {output.shape}")

# ── 4. Comparaison paramètres ──
print("\\n═══ Paramètres ═══")
for name, m in [('RNN', rnn), ('LSTM', lstm), ('GRU', gru)]:
    p = sum(p.numel() for p in m.parameters())
    print(f"  {name:>4s}: {p:,} params")
print("  LSTM ≈ 4× RNN (4 portes), GRU ≈ 3× RNN (3 gates)")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 12 — TRANSFORMERS (Ch. 12-13)
  // ═══════════════════════════════════════
  {
    id: 'attention',
    title: 'Attention & Transformers',
    shortTitle: 'Transformer',
    description: 'Self-Attention, Multi-Head Attention, Positional Encoding, blocs Transformer encoder/decoder (Ch. 12-13 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['rnn'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `## 12.1 — Du RNN au Transformer\n\nLes RNN traitent les tokens **séquentiellement** — impossible de paralléliser, et l'information se dégrade sur les longues séquences. Le **Transformer** (Vaswani et al., 2017) remplace la récurrence par le **self-attention** : chaque token consulte directement tous les autres en parallèle.\n\nRésultat : parallélisme massif + pas de vanishing gradient temporel.`,
      },
      {
        type: 'text',
        content: `## 12.2 — Dot-Product Self-Attention\n\nPour chaque entrée xₘ, on projette vers 3 vecteurs :\n- **Query** q = Wq·x : "quelle information je cherche"\n- **Key** k = Wk·x : "quelle information j'offre"\n- **Value** v = Wv·x : "le contenu à transmettre"\n\nLa sortie de l'attention est une somme pondérée des values, avec des poids calculés par compatibilité query-key :`,
      },
      {
        type: 'equation',
        content: '\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\!\\left(\\frac{\\mathbf{Q}\\mathbf{K}^\\top}{\\sqrt{d_k}}\\right) \\mathbf{V}',
        label: 'Éq. 12.14 — Scaled Dot-Product Attention',
        highlightVar: 'attention',
      },
      {
        type: 'diagram',
        content: `  Scaled Dot-Product Attention
  ─────────────────────────────
      Q    K    V
      │    │    │
      └──┬─┘    │
         │      │
     Q·K^T/√dk  │
         │      │
      softmax   │
         │      │
         └──┬───┘
            │
        Attn @ V
            │
         Output
  
  Complexité : O(N² · d)  — quadratique en longueur de séquence`,
        label: 'Fig. 12.3 — Attention Pipeline',
      },
      {
        type: 'text',
        content: `## 12.3 — Multi-Head Attention\n\nAu lieu d'un seul mécanisme d'attention, on exécute **H têtes** en parallèle. Chaque tête utilise des projections différentes (Wq_h, Wk_h, Wv_h) et apprend des types de relations distincts : syntaxe, coréférence, sémantique, etc.\n\nLes sorties des H têtes sont concaténées puis projetées par Wo :`,
      },
      {
        type: 'equation',
        content: '\\text{MultiHead}(\\mathbf{X}) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_H) \\, \\mathbf{W}_O \\quad \\text{où}\\; \\text{head}_h = \\text{Attn}(\\mathbf{X}\\mathbf{W}_h^Q, \\mathbf{X}\\mathbf{W}_h^K, \\mathbf{X}\\mathbf{W}_h^V)',
        label: 'Éq. 12.18 — Multi-Head Attention',
      },
      {
        type: 'text',
        content: `## 12.4 — Positional Encoding\n\nLe self-attention est **permutation-invariant** — il ne connaît pas l'ordre des tokens ! On ajoute un **positional encoding** pour injecter la notion de position :\n\n**Sinusoidal** (Vaswani) : PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))\n\n**Appris** (BERT, GPT) : une matrice (max_len × d_model) entraînable.`,
      },
      {
        type: 'text',
        content: `## 12.5 — Bloc Transformer\n\nUn bloc Transformer empile :\n1. **Multi-Head Self-Attention** + résiduel + LayerNorm\n2. **Feed-Forward Network** (2 couches, ReLU/GELU) + résiduel + LayerNorm\n\nOn empile L blocs identiques. BERT-base : L=12, d=768, H=12. GPT-3 : L=96, d=12288, H=96.`,
      },
      {
        type: 'diagram',
        content: `  Transformer Block (Pre-Norm variant)
  ╔═══════════════════════════════════╗
  ║  Input x                         ║
  ║    │                              ║
  ║    ├───────────────────┐          ║
  ║    ▼                   │          ║
  ║  LayerNorm             │          ║
  ║    ▼                   │          ║
  ║  Multi-Head Attention  │          ║
  ║    ▼                   │          ║
  ║    + ◄─────────────────┘ (résiduel)║
  ║    │                              ║
  ║    ├───────────────────┐          ║
  ║    ▼                   │          ║
  ║  LayerNorm             │          ║
  ║    ▼                   │          ║
  ║  FFN (Linear→GELU→Linear)        ║
  ║    ▼                   │          ║
  ║    + ◄─────────────────┘ (résiduel)║
  ║    │                              ║
  ║  Output                          ║
  ╚═══════════════════════════════════╝`,
        label: 'Fig. 12.7 — Bloc Transformer',
      },
      {
        type: 'text',
        content: `## 12.6 — Encoder vs Decoder\n\n- **Encoder** (BERT) : self-attention **bidirectionnelle** — chaque token voit tous les autres. Pour la compréhension (classification, NER).\n- **Decoder** (GPT) : self-attention **causale** — un masque triangulaire empêche de voir les tokens futurs. Pour la génération.\n- **Encoder-Decoder** (T5, traduction) : le decoder utilise le **cross-attention** pour consulter l'encoder.`,
      },
      {
        type: 'text',
        content: `## 12.7 — Masque Causal\n\nPour la génération autoregressive (GPT), on applique un masque **triangulaire inférieur** avant le softmax. Les positions futures sont remplies de −∞, ce qui donne un poids d'attention de 0 après softmax. Le token n ne peut voir que les tokens 1..n.`,
      },
      {
        type: 'callout',
        content: '🧠 **Hypernetwork** : le self-attention est un réseau dont les poids (attention scores) sont eux-mêmes calculés par le réseau. C\'est ce qui le rend si flexible — les connexions **dépendent des données**.\n\n📊 **Échelle** : BERT-base = 110M params, GPT-2 = 1.5B, GPT-3 = 175B, GPT-4 ~ 1.8T (estimé). La loi d\'échelle montre que la performance s\'améliore en power-law avec la taille.',
      },
    ],
    exercises: [
      {
        id: 'attn-ex1',
        title: '💻 Pratique — Self-Attention from scratch',
        instructions: 'Implémentez le scaled dot-product attention manuellement (sans nn.MultiheadAttention), puis ajoutez le masque causal.',
        starterCode: `import torch
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

# ── 1. Attention bidirectionnelle ──
scores = ___  # Q @ K^T / sqrt(d_k)
weights = ___  # softmax(scores)
output = ___   # weights @ V
print(f"Bidirectional output: {output.shape}")

# ── 2. Attention causale (masque triangulaire) ──
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores_causal = scores.masked_fill(mask, float('-inf'))
weights_causal = torch.softmax(scores_causal, dim=-1)
output_causal = weights_causal @ V
print(f"Causal output: {output_causal.shape}")
print(f"Causal weights row 0: {weights_causal[0, 0]}")  # seul le 1er token a un poids`,
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

# ── 1. Attention bidirectionnelle ──
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)
weights = torch.softmax(scores, dim=-1)
output = weights @ V
print(f"Bidirectional output: {output.shape}")

# ── 2. Attention causale ──
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores_causal = scores.masked_fill(mask, float('-inf'))
weights_causal = torch.softmax(scores_causal, dim=-1)
output_causal = weights_causal @ V
print(f"Causal output: {output_causal.shape}")
print(f"Causal weights row 0: {weights_causal[0, 0]}")`,
        hints: [
          'scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)',
          'Le masque causal met -inf aux positions futures, le softmax les transforme en 0',
        ],
        completed: false,
      },
      {
        id: 'attn-ex2',
        title: '💻 Pratique — Transformer Block complet',
        instructions: 'Construisez un bloc Transformer complet avec Multi-Head Attention, FFN, résiduel et LayerNorm.',
        starterCode: `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, dk)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x, mask=None):
        # Pre-norm: x + Attn(LN(x))
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

# Test
block = TransformerBlock(d_model=128, n_heads=8)
x = torch.randn(2, 10, 128)
out = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in block.parameters()):,}")
print(f"  - Attn: {sum(p.numel() for p in block.attn.parameters()):,}")
print(f"  - FFN:  {sum(p.numel() for p in block.ffn.parameters()):,}")`,
        solution: `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

block = TransformerBlock(d_model=128, n_heads=8)
x = torch.randn(2, 10, 128)
out = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in block.parameters()):,}")
print(f"  - Attn: {sum(p.numel() for p in block.attn.parameters()):,}")
print(f"  - FFN:  {sum(p.numel() for p in block.ffn.parameters()):,}")`,
        hints: [
          'Pre-norm : LN avant attention et FFN, pas après',
          'W_qkv projette vers 3×d_model, puis on split en Q, K, V',
          'd_ff = 4 × d_model est le standard',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn
import math

# ══════════════════════════════════════════════════════════════
# Attention & Transformers — Ch. 12-13 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Scaled Dot-Product Attention ──
print("═══ Scaled Dot-Product Attention ═══")
d_model = 64
seq_len = 8
x = torch.randn(1, seq_len, d_model)

W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

Q, K, V = W_q(x), W_k(x), W_v(x)
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)
attn_weights = torch.softmax(scores, dim=-1)
output = attn_weights @ V
print(f"Attention weights: {attn_weights.shape}")
print(f"Output: {output.shape}")

# ── 2. Masque causal (GPT-style) ──
print("\\n═══ Causal Mask ═══")
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores_masked = scores.masked_fill(causal_mask, float('-inf'))
attn_causal = torch.softmax(scores_masked, dim=-1)
print(f"Causal attn row 0: {attn_causal[0, 0].tolist()}")
print(f"Causal attn row 4: {attn_causal[0, 4].tolist()}")

# ── 3. Multi-Head Attention ──
print("\\n═══ Multi-Head Attention ═══")
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
out, weights = mha(x, x, x)
print(f"MHA output: {out.shape}")
print(f"MHA weights: {weights.shape}")

# ── 4. Positional Encoding ──
print("\\n═══ Positional Encoding ═══")
max_len = 100
pe = torch.zeros(max_len, d_model)
pos = torch.arange(0, max_len).unsqueeze(1).float()
div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
print(f"PE shape: {pe.shape}")
print(f"PE[0, :8]: {pe[0, :8].tolist()}")
print(f"PE[1, :8]: {pe[1, :8].tolist()}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 13 — GANs (Ch. 15)
  // ═══════════════════════════════════════
  {
    id: 'gan',
    title: 'Generative Adversarial Networks (GAN)',
    shortTitle: 'GAN',
    description: 'Objectif minimax, entraînement adversarial, mode collapse, WGAN, et astuces pratiques (Ch. 15 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['attention'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `## 15.1 — Le principe adversarial\n\nUn **GAN** met en compétition deux réseaux :\n\n- **Générateur G(z)** : transforme du bruit z ~ N(0,I) en données synthétiques\n- **Discriminateur D(x)** : estime P(x est réel)\n\nG cherche à **tromper** D. D cherche à **ne pas être trompé**. Ce **jeu minimax** produit un équilibre de Nash où G génère des données indistinguables des réelles.`,
      },
      {
        type: 'equation',
        content: '\\min_G \\max_D \\; V(D,G) = \\mathbb{E}_{\\mathbf{x} \\sim p_{\\text{data}}}[\\log D(\\mathbf{x})] + \\mathbb{E}_{\\mathbf{z} \\sim p_z}[\\log(1 - D(G(\\mathbf{z}))))]',
        label: 'Éq. 15.1 — Objectif Minimax du GAN',
      },
      {
        type: 'text',
        content: `## 15.2 — Algorithme d'entraînement\n\nÀ chaque itération :\n\n**Étape 1 — Entraîner D** (k pas) :\n- Échantillonner un mini-batch réel x et un bruit z\n- Calculer loss_D = −[log D(x) + log(1 − D(G(z)))]\n- Mettre à jour D par gradient ascent\n\n**Étape 2 — Entraîner G** (1 pas) :\n- Échantillonner un bruit z\n- Calculer loss_G = −log D(G(z))  ← "non-saturating" trick\n- Mettre à jour G par gradient descent\n\n⚠️ En pratique, on minimise −log D(G(z)) au lieu de log(1−D(G(z))) pour éviter le vanishing gradient quand G est mauvais.`,
      },
      {
        type: 'diagram',
        content: `  Architecture GAN
  ──────────────────────────────────
       z ~ N(0,I)      x ~ p_data
          │                │
          ▼                │
    ┌───────────┐          │
    │ Générateur│          │
    │     G     │          │
    └─────┬─────┘          │
          │                │
       G(z)               x
          │                │
          └───────┬────────┘
                  ▼
          ┌──────────────┐
          │Discriminateur│
          │      D       │
          └──────┬───────┘
                 │
          D(·) ∈ [0,1]
          0 = faux, 1 = vrai`,
        label: 'Fig. 15.2 — Architecture GAN',
      },
      {
        type: 'text',
        content: `## 15.3 — Mode Collapse\n\nLe problème le plus courant des GANs : G apprend à générer seulement **quelques modes** de la distribution au lieu de la distribution complète. Par exemple, G ne produit que des "7" au lieu de tous les chiffres.\n\n**Causes** : G trouve un point fixe qui trompe D systématiquement, donc il n'a pas d'incitation à diversifier.\n\n**Solutions** : mini-batch discrimination, unrolled GAN, feature matching.`,
      },
      {
        type: 'text',
        content: `## 15.4 — WGAN (Wasserstein GAN)\n\nRemplace la divergence JS par la **distance de Wasserstein** (Earth Mover's Distance), ce qui donne des gradients plus stables et un signal même quand les distributions ne se chevauchent pas.\n\nLe discriminateur devient un **critique** (pas de sigmoid) avec contrainte de Lipschitz :\n- **WGAN** : weight clipping\n- **WGAN-GP** : gradient penalty (λ·(||∇D(x̂)||₂ − 1)²)`,
      },
      {
        type: 'equation',
        content: '\\min_G \\max_{D \\in \\mathcal{D}_L} \\; \\mathbb{E}_{\\mathbf{x}}[D(\\mathbf{x})] - \\mathbb{E}_{\\mathbf{z}}[D(G(\\mathbf{z}))]',
        label: 'Éq. 15.8 — Objectif WGAN',
      },
      {
        type: 'callout',
        content: '⚡ **Astuces d\'entraînement GAN** :\n• Utiliser \\`LeakyReLU(0.2)\\` dans D (pas ReLU)\n• \\`BatchNorm\\` dans G (pas dans D — ou spectral norm)\n• Labels lissés : 0.9 au lieu de 1.0 pour les vrais\n• Adam avec lr=0.0002, betas=(0.5, 0.999)\n• Entraîner D plus souvent que G (k=5 pour WGAN)\n• Architectures notables : DCGAN, StyleGAN, StyleGAN2, StyleGAN3',
      },
    ],
    exercises: [
      {
        id: 'gan-ex1',
        title: '💻 Pratique — GAN simple sur 2D',
        instructions: 'Implémentez un GAN qui apprend à générer des points sur un cercle (distribution 2D simple).',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

# ── Distribution cible : points sur un cercle ──
def sample_circle(n, noise=0.05):
    theta = torch.rand(n) * 2 * 3.14159
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x + torch.randn_like(x) * noise

# ── Générateur ──
G = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)

# ── Discriminateur ──
D = nn.Sequential(
    nn.Linear(2, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ── Entraînement ──
for epoch in range(2000):
    # 1. Entraîner D
    real = sample_circle(128)
    z = torch.randn(128, 2)
    fake = G(z).detach()
    
    loss_D = criterion(D(real), torch.ones(128, 1)) + \\
             criterion(D(fake), torch.zeros(128, 1))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()
    
    # 2. Entraîner G
    z = torch.randn(128, 2)
    fake = G(z)
    loss_G = criterion(D(fake), torch.ones(128, 1))  # tromper D
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | D loss: {loss_D.item():.4f} | G loss: {loss_G.item():.4f}")

# ── Résultat ──
z = torch.randn(500, 2)
generated = G(z).detach()
print(f"\\nMoyenne rayon généré: {generated.norm(dim=1).mean():.3f} (cible ≈ 1.0)")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

def sample_circle(n, noise=0.05):
    theta = torch.rand(n) * 2 * 3.14159
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x + torch.randn_like(x) * noise

G = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
)

D = nn.Sequential(
    nn.Linear(2, 64), nn.LeakyReLU(0.2),
    nn.Linear(64, 64), nn.LeakyReLU(0.2),
    nn.Linear(64, 1), nn.Sigmoid(),
)

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(2000):
    real = sample_circle(128)
    z = torch.randn(128, 2)
    fake = G(z).detach()
    
    loss_D = criterion(D(real), torch.ones(128, 1)) + \\
             criterion(D(fake), torch.zeros(128, 1))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()
    
    z = torch.randn(128, 2)
    fake = G(z)
    loss_G = criterion(D(fake), torch.ones(128, 1))
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | D loss: {loss_D.item():.4f} | G loss: {loss_G.item():.4f}")

z = torch.randn(500, 2)
generated = G(z).detach()
print(f"\\nMoyenne rayon généré: {generated.norm(dim=1).mean():.3f} (cible ≈ 1.0)")`,
        hints: [
          'G(z).detach() empêche les gradients de G de fuiter dans D',
          'loss_G utilise torch.ones — on veut que D dise "vrai" pour les faux',
        ],
        completed: false,
      },
      {
        id: 'gan-th1',
        title: '🧠 Théorie — Comparer GAN vs WGAN',
        instructions: 'Implémentez un WGAN-GP (Wasserstein GAN with Gradient Penalty) et comparez la stabilité avec un GAN classique.',
        starterCode: `import torch
import torch.nn as nn

# ── Gradient Penalty (WGAN-GP) ──
def gradient_penalty(D, real, fake, lambda_gp=10.0):
    """Calcule la pénalité de gradient pour WGAN-GP"""
    alpha = torch.rand(real.size(0), 1)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out = D(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

print("💡 WGAN-GP : le critique (D sans sigmoid) est contraint")
print("   à être 1-Lipschitz via la pénalité de gradient")
print("   → gradients stables, pas de mode collapse")`,
        solution: `import torch
import torch.nn as nn

def gradient_penalty(D, real, fake, lambda_gp=10.0):
    alpha = torch.rand(real.size(0), 1)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out = D(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

print("WGAN-GP : le critique est contraint à être 1-Lipschitz via gradient penalty")`,
        hints: [
          'Le critique WGAN n\'a PAS de sigmoid — sortie non bornée',
          'La gradient penalty interpole entre vrais et faux pour contraindre ||∇D|| ≈ 1',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Generative Adversarial Networks — Ch. 15 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Architecture GAN de base ──
print("═══ GAN Architecture ═══")

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()  # sortie dans [-1, 1]
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # P(réel)
        )
    
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()

z = torch.randn(4, 100)
fake = G(z)
score = D(fake)

print(f"Bruit z:        {z.shape}")
print(f"Image générée:  {fake.shape}")
print(f"Score D(G(z)):  {score.squeeze().tolist()}")
print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

# ── 2. Losses ──
print("\\n═══ GAN Losses ═══")
criterion = nn.BCELoss()
real_data = torch.randn(4, 784)

# D loss
loss_real = criterion(D(real_data), torch.ones(4, 1))
loss_fake = criterion(D(G(z).detach()), torch.zeros(4, 1))
loss_D = loss_real + loss_fake
print(f"D loss: {loss_D.item():.4f}")

# G loss (non-saturating)
loss_G = criterion(D(G(z)), torch.ones(4, 1))
print(f"G loss: {loss_G.item():.4f}")
`,
  },

  // ═══════════════════════════════════════
  // MODULE 14 — DIFFUSION MODELS (Ch. 18)
  // ═══════════════════════════════════════
  {
    id: 'diffusion',
    title: 'Modèles de Diffusion',
    shortTitle: 'Diffusion',
    description: 'Forward/reverse process, DDPM, reparameterization trick, U-Net denoiser, classifier-free guidance (Ch. 18 — UDL).',
    status: 'locked',
    progress: 0,
    dependencies: ['attention'],
    category: 'advanced',
    theory: [
      {
        type: 'text',
        content: `## 18.1 — L'idée de la diffusion\n\nLes **modèles de diffusion** apprennent à générer des données en inversant un processus de bruitage progressif en T étapes (typiquement T=1000).\n\n- **Forward process** q(xₜ|xₜ₋₁) : ajouter un petit peu de bruit gaussien à chaque étape. Au bout de T étapes, x_T ≈ bruit pur N(0,I).\n- **Reverse process** p_θ(xₜ₋₁|xₜ) : un réseau neuronal apprend à **débruiter** — retirer le bruit étape par étape pour reconstruire l'image.`,
      },
      {
        type: 'equation',
        content: 'q(\\mathbf{x}_t | \\mathbf{x}_{t-1}) = \\mathcal{N}\\big(\\mathbf{x}_t;\\; \\sqrt{1-\\beta_t}\\,\\mathbf{x}_{t-1},\\; \\beta_t \\mathbf{I}\\big)',
        label: 'Éq. 18.1 — Forward Process (1 étape)',
      },
      {
        type: 'text',
        content: `## 18.2 — Reparameterization trick\n\nGrâce à la propriété d'additivité des gaussiennes, on peut **sauter directement** à n'importe quel timestep t sans itérer :\n\nEn posant αₜ = 1−βₜ et ᾱₜ = ∏ₛ₌₁ᵗ αₛ, on obtient :`,
      },
      {
        type: 'equation',
        content: '\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})',
        label: 'Éq. 18.5 — Closed-form sampling',
        highlightVar: 'x_t',
      },
      {
        type: 'diagram',
        content: `  Processus de Diffusion (DDPM)
  ═══════════════════════════════════════════
  
  Forward (bruitage progressif) →
  ┌─────┐   ┌─────┐   ┌─────┐       ┌─────┐
  │ x₀  │──▶│ x₁  │──▶│ x₂  │──▶···│ x_T │
  │image│   │     │   │     │       │bruit│
  └─────┘   └─────┘   └─────┘       └─────┘
                                        │
  ← Reverse (débruitage appris)         │
  ┌─────┐   ┌─────┐   ┌─────┐       ┌──┴──┐
  │ x₀  │◀──│ x₁  │◀──│ x₂  │◀──···│ x_T │
  │image│   │     │   │     │       │bruit│
  └─────┘   └─────┘   └─────┘       └─────┘
     ▲          ▲          ▲
     └─ εθ(xₜ,t) prédit le bruit à chaque étape`,
        label: 'Fig. 18.1 — Forward et Reverse Process',
      },
      {
        type: 'text',
        content: `## 18.3 — Objectif d'entraînement (DDPM)\n\nLe réseau εθ apprend à **prédire le bruit** ε ajouté à x₀ pour obtenir xₜ. L'objectif simplifié de Ho et al. (2020) est une simple MSE :`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t \\sim U(1,T),\\; \\mathbf{x}_0,\\; \\boldsymbol{\\epsilon}} \\Big[ \\big\\| \\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t) \\big\\|^2 \\Big]',
        label: 'Éq. 18.10 — Objectif simplifié DDPM',
        highlightVar: 'loss',
      },
      {
        type: 'text',
        content: `## 18.4 — Architecture U-Net\n\nLe denoiser εθ est typiquement un **U-Net** : un réseau encodeur-décodeur avec des **skip connections** entre couches de même résolution. Le timestep t est injecté via un **embedding sinusoidal** (comme le positional encoding des Transformers).\n\n**Stable Diffusion** ajoute du **cross-attention** dans le U-Net pour conditionner la génération sur du texte (CLIP embeddings).`,
      },
      {
        type: 'text',
        content: `## 18.5 — Sampling (génération)\n\n**Algorithme DDPM Sampling** :\n1. Échantillonner x_T ~ N(0, I)\n2. Pour t = T, T−1, ..., 1 :\n   a. Prédire le bruit : ε̂ = εθ(xₜ, t)\n   b. Calculer xₜ₋₁ = (1/√αₜ)(xₜ − (βₜ/√(1−ᾱₜ))ε̂) + σₜz\n3. Retourner x₀\n\n⚠️ 1000 étapes de débruitage = lent → **DDIM** (50 étapes), **DPM-Solver** (20 étapes).`,
      },
      {
        type: 'text',
        content: `## 18.6 — Classifier-Free Guidance\n\nPour contrôler la génération avec un **prompt texte**, on entraîne le même modèle avec et sans condition (en dropout du prompt avec probabilité p=0.1). À l'inférence, on amplifie la direction conditionnelle :\n\nε̂ = εθ(xₜ, ∅) + w · (εθ(xₜ, c) − εθ(xₜ, ∅))\n\nw > 1 (typiquement 7.5) renforce l'adhérence au prompt au détriment de la diversité.`,
      },
      {
        type: 'callout',
        content: '🧠 **Modèles notables** :\n• **DDPM** (Ho 2020) : l\'article fondateur\n• **DALL-E 2** (OpenAI) : diffusion + CLIP\n• **Stable Diffusion** (Stability AI) : diffusion dans l\'espace latent (LDM)\n• **Midjourney** : variante propriétaire\n• **Imagen** (Google) : cascaded diffusion\n• **FLUX, SD3** : architectures DiT (Diffusion Transformer) — remplacent le U-Net par des Transformers',
      },
    ],
    exercises: [
      {
        id: 'diff-ex1',
        title: '💻 Pratique — Forward Process DDPM',
        instructions: 'Implémentez le forward process et visualisez comment une image se dégrade progressivement avec le bruit.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

# ── Schedule de bruit ──
T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# ── Forward process : sauter directement au timestep t ──
def forward_diffusion(x0, t, noise=None):
    """Ajoute du bruit au timestep t (closed-form)"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1, 1)
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t]).view(-1, 1)
    return sqrt_ab * x0 + sqrt_1_ab * noise, noise

# ── Test sur un "signal" 1D ──
x0 = torch.sin(torch.linspace(0, 4 * 3.14159, 100)).unsqueeze(0)  # signal sinusoïdal

print(f"{'t':>6} │ {'ᾱₜ':>8} │ {'SNR (dB)':>10} │ {'x_t std':>8}")
print(f"{'─'*6}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*8}")

for t_val in [0, 50, 100, 250, 500, 750, 999]:
    t = torch.tensor([t_val])
    x_t, eps = forward_diffusion(x0, t)
    ab = alpha_bar[t_val].item()
    snr = 10 * torch.log10(torch.tensor(ab / (1 - ab))).item()
    print(f"{t_val:6d} │ {ab:8.4f} │ {snr:10.2f} │ {x_t.std():8.4f}")

print(f"\\n→ À t=0, signal intact (ᾱ≈1)")
print(f"→ À t=999, bruit pur (ᾱ≈0)")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def forward_diffusion(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1, 1)
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t]).view(-1, 1)
    return sqrt_ab * x0 + sqrt_1_ab * noise, noise

x0 = torch.sin(torch.linspace(0, 4 * 3.14159, 100)).unsqueeze(0)

print(f"{'t':>6} │ {'ᾱₜ':>8} │ {'SNR (dB)':>10} │ {'x_t std':>8}")
print(f"{'─'*6}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*8}")

for t_val in [0, 50, 100, 250, 500, 750, 999]:
    t = torch.tensor([t_val])
    x_t, eps = forward_diffusion(x0, t)
    ab = alpha_bar[t_val].item()
    snr = 10 * torch.log10(torch.tensor(ab / (1 - ab))).item()
    print(f"{t_val:6d} │ {ab:8.4f} │ {snr:10.2f} │ {x_t.std():8.4f}")

print(f"\\n→ À t=0, signal intact (ᾱ≈1)")
print(f"→ À t=999, bruit pur (ᾱ≈0)")`,
        hints: [
          'alpha_bar = cumprod des (1-beta)',
          'x_t = sqrt(ᾱₜ)·x₀ + sqrt(1−ᾱₜ)·ε',
          'Le SNR décroît monotoniquement avec t',
        ],
        completed: false,
      },
      {
        id: 'diff-ex2',
        title: '💻 Pratique — Denoiser simple + training loop',
        instructions: 'Entraînez un petit denoiser MLP sur des données 1D pour comprendre la boucle DDPM.',
        starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)

# ── Schedule ──
T = 200
betas = torch.linspace(0.001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# ── Données : distribution bimodale 1D ──
def sample_data(n):
    mix = torch.rand(n) > 0.5
    return mix.float() * 2 - 1 + torch.randn(n) * 0.1  # pics à -1 et +1

# ── Denoiser ──
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),   # x_t + t_embed
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x_t, t_norm):
        inp = torch.stack([x_t, t_norm], dim=-1)
        return self.net(inp).squeeze(-1)

model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training ──
for step in range(3000):
    x0 = sample_data(256)
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    
    x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * eps
    eps_pred = model(x_t, t.float() / T)
    
    loss = nn.MSELoss()(eps_pred, eps)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (step + 1) % 1000 == 0:
        print(f"Step {step+1:4d} | Loss: {loss.item():.4f}")

# ── Sampling ──
print("\\n═══ Sampling ═══")
x = torch.randn(1000)
for t in reversed(range(T)):
    t_batch = torch.full((1000,), t)
    eps_pred = model(x, t_batch.float() / T)
    
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]
    x = (1 / torch.sqrt(alpha_t)) * (x - (betas[t] / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
    if t > 0:
        x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

print(f"Échantillons générés — mean: {x.mean():.3f}, std: {x.std():.3f}")
print(f"Modes attendus: -1 et +1")
print(f"Fraction < 0: {(x < 0).float().mean():.2f} (attendu ≈ 0.50)")`,
        solution: `import torch
import torch.nn as nn

torch.manual_seed(42)

T = 200
betas = torch.linspace(0.001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def sample_data(n):
    mix = torch.rand(n) > 0.5
    return mix.float() * 2 - 1 + torch.randn(n) * 0.1

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x_t, t_norm):
        inp = torch.stack([x_t, t_norm], dim=-1)
        return self.net(inp).squeeze(-1)

model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(3000):
    x0 = sample_data(256)
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    
    x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * eps
    eps_pred = model(x_t, t.float() / T)
    
    loss = nn.MSELoss()(eps_pred, eps)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (step + 1) % 1000 == 0:
        print(f"Step {step+1:4d} | Loss: {loss.item():.4f}")

print("\\n═══ Sampling ═══")
x = torch.randn(1000)
for t in reversed(range(T)):
    t_batch = torch.full((1000,), t)
    eps_pred = model(x, t_batch.float() / T)
    
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]
    x = (1 / torch.sqrt(alpha_t)) * (x - (betas[t] / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
    if t > 0:
        x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

print(f"Échantillons générés — mean: {x.mean():.3f}, std: {x.std():.3f}")
print(f"Modes attendus: -1 et +1")
print(f"Fraction < 0: {(x < 0).float().mean():.2f} (attendu ≈ 0.50)")`,
        hints: [
          'x_t = sqrt(ᾱₜ)·x₀ + sqrt(1−ᾱₜ)·ε — le forward process',
          'Le sampling inverse la formule étape par étape',
          'Le denoiser reçoit x_t et t normalisé comme entrées',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# Modèles de Diffusion — Ch. 18 Understanding Deep Learning
# ══════════════════════════════════════════════════════════════

# ── 1. Noise Schedule ──
print("═══ Noise Schedule DDPM ═══")
T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

print(f"β_1 = {betas[0]:.4f}, β_T = {betas[-1]:.4f}")
print(f"ᾱ_1 = {alpha_bar[0]:.4f}, ᾱ_T = {alpha_bar[-1]:.6f}")

# ── 2. Forward Process ──
print("\\n═══ Forward Process ═══")
x0 = torch.randn(1, 784)  # image "propre"

def add_noise(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1, 1)
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t]).view(-1, 1)
    return sqrt_ab * x0 + sqrt_1_ab * noise, noise

for t_val in [0, 100, 500, 999]:
    t = torch.tensor([t_val])
    x_t, _ = add_noise(x0, t)
    print(f"  t={t_val:4d}: ᾱ={alpha_bar[t_val]:.4f}, ||x_t||={x_t.norm():.2f}")

# ── 3. Simple Denoiser ──
print("\\n═══ Denoiser Architecture ═══")
class SimpleDenoiser(nn.Module):
    def __init__(self, dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, x_noisy, t_norm):
        inp = torch.cat([x_noisy, t_norm.unsqueeze(-1)], dim=-1)
        return self.net(inp)

model = SimpleDenoiser()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ── 4. Training step ──
print("\\n═══ Training Step ═══")
t = torch.randint(0, T, (4,))
eps = torch.randn(4, 784)
x_t = torch.sqrt(alpha_bar[t]).unsqueeze(1) * x0 + torch.sqrt(1 - alpha_bar[t]).unsqueeze(1) * eps
eps_pred = model(x_t, t.float() / T)
loss = nn.MSELoss()(eps_pred, eps)
print(f"Loss (untrained): {loss.item():.4f}")
print(f"Objectif: prédire le bruit ε ajouté à x₀")
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
