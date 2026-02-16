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
  // ═══════════════════════════════════════
  {
    id: 'shallow-networks',
    title: 'Réseaux de Neurones Superficiels',
    shortTitle: 'Shallow Net',
    description: 'Comprendre les réseaux à une couche cachée, ReLU, et le théorème d\'approximation universelle.',
    status: 'available',
    progress: 0,
    dependencies: ['tensors'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Un **réseau de neurones superficiel** (shallow network) est une fonction **y = f[x, ϕ]** avec une seule couche cachée. Il prend une entrée, calcule des **unités cachées** (hidden units) via une activation, puis combine linéairement ces unités pour produire la sortie.\n\nLe réseau se décompose en trois étapes :\n1. Calculer des fonctions linéaires de l'entrée\n2. Appliquer une **fonction d'activation** a[•]\n3. Combiner linéairement les résultats`,
      },
      {
        type: 'equation',
        content: 'y = \\phi_0 + \\phi_1 \\, a[\\theta_{10} + \\theta_{11} x] + \\phi_2 \\, a[\\theta_{20} + \\theta_{21} x] + \\phi_3 \\, a[\\theta_{30} + \\theta_{31} x]',
        label: 'Réseau superficiel (Shallow Network)',
        highlightVar: 'output',
      },
      {
        type: 'text',
        content: `La fonction d'activation la plus utilisée est le **ReLU** (Rectified Linear Unit). Elle retourne l'entrée si elle est positive, et zéro sinon. Cette simplicité rend le calcul efficace et produit des **fonctions linéaires par morceaux**.`,
      },
      {
        type: 'equation',
        content: 'a[z] = \\text{ReLU}[z] = \\begin{cases} 0 & \\text{si } z < 0 \\\\ z & \\text{si } z \\geq 0 \\end{cases}',
        label: 'Rectified Linear Unit (ReLU)',
        highlightVar: 'relu',
      },
      {
        type: 'callout',
        content: '🧠 **Théorème d\'approximation universelle** : un réseau superficiel avec suffisamment d\'unités cachées peut approximer n\'importe quelle fonction continue sur un compact. Chaque unité cachée contribue un "joint" à la fonction, créant des régions linéaires supplémentaires.',
      },
      {
        type: 'text',
        content: `Les **unités cachées** h₁, h₂, h₃ sont des résultats intermédiaires. Chaque unité contient une fonction linéaire de l'entrée, clippée par ReLU. La sortie finale est une combinaison linéaire de ces unités :\n\n**y = ϕ₀ + ϕ₁h₁ + ϕ₂h₂ + ϕ₃h₃**\n\nAvec D unités cachées, on obtient un maximum de D+1 régions linéaires.`,
      },
    ],
    exercises: [
      {
        id: 'shallow-ex1',
        title: 'Un réseau à une couche cachée',
        instructions: 'Implémentez un réseau superficiel avec 3 unités cachées et activation ReLU. Calculez la sortie pour une entrée x = 1.5.',
        starterCode: `import torch

def relu(z):
    return ___

x = torch.tensor(1.5)

# Paramètres de la couche cachée
theta = torch.tensor([[0.5, -1.0],   # theta_10, theta_11
                       [-0.3, 0.8],   # theta_20, theta_21
                       [0.1, 1.2]])   # theta_30, theta_31

# Paramètres de sortie
phi = torch.tensor([0.2, 0.5, -0.3, 0.7])  # phi_0, phi_1, phi_2, phi_3

# Calcul des unités cachées
h1 = relu(___)
h2 = relu(___)
h3 = relu(___)

# Sortie
y = ___

print(f"h1={h1.item():.4f}, h2={h2.item():.4f}, h3={h3.item():.4f}")
print(f"y = {y.item():.4f}")`,
        solution: `import torch

def relu(z):
    return torch.clamp(z, min=0)

x = torch.tensor(1.5)

# Paramètres de la couche cachée
theta = torch.tensor([[0.5, -1.0],
                       [-0.3, 0.8],
                       [0.1, 1.2]])

# Paramètres de sortie
phi = torch.tensor([0.2, 0.5, -0.3, 0.7])

# Calcul des unités cachées
h1 = relu(theta[0, 0] + theta[0, 1] * x)
h2 = relu(theta[1, 0] + theta[1, 1] * x)
h3 = relu(theta[2, 0] + theta[2, 1] * x)

# Sortie
y = phi[0] + phi[1] * h1 + phi[2] * h2 + phi[3] * h3

print(f"h1={h1.item():.4f}, h2={h2.item():.4f}, h3={h3.item():.4f}")
print(f"y = {y.item():.4f}")`,
        hints: [
          'relu(z) = torch.clamp(z, min=0) ou torch.max(z, torch.tensor(0.0))',
          'h1 = relu(theta[0,0] + theta[0,1] * x)',
          'y = phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ══ Réseau Superficiel (Shallow Network) ══
# Implémentation selon le livre "Understanding Deep Learning"

def shallow_network(x, theta, phi):
    """
    Réseau à 1 couche cachée avec 3 unités + ReLU
    x: entrée scalaire
    theta: paramètres couche cachée (3x2)
    phi: paramètres sortie (4,)
    """
    # Unités cachées avec ReLU
    h1 = torch.relu(theta[0, 0] + theta[0, 1] * x)
    h2 = torch.relu(theta[1, 0] + theta[1, 1] * x)
    h3 = torch.relu(theta[2, 0] + theta[2, 1] * x)
    
    # Combinaison linéaire
    y = phi[0] + phi[1] * h1 + phi[2] * h2 + phi[3] * h3
    return y

# Paramètres
theta = torch.tensor([[0.5, -1.0], [-0.3, 0.8], [0.1, 1.2]])
phi = torch.tensor([0.2, 0.5, -0.3, 0.7])

# Test sur plusieurs entrées
for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    x = torch.tensor(x_val)
    y = shallow_network(x, theta, phi)
    print(f"f({x_val:+.1f}) = {y.item():.4f}")

# ── Avec PyTorch nn.Module ──
model = nn.Sequential(
    nn.Linear(1, 10),    # 10 hidden units
    nn.ReLU(),
    nn.Linear(10, 1)
)
print(f"\\nParamètres: {sum(p.numel() for p in model.parameters())}")
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
