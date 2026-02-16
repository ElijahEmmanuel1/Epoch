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
export const courseNodes: CourseNode[] = [
  {
    id: 'tensors',
    title: 'Tenseurs & Opérations',
    shortTitle: 'Tenseurs',
    description: 'Comprendre les tenseurs, leur forme et les opérations fondamentales en PyTorch.',
    status: 'available',
    progress: 0,
    dependencies: [],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Un **tenseur** est la structure de données fondamentale du Deep Learning. C'est une généralisation des matrices à N dimensions.\n\n- Un **scalaire** est un tenseur de rang 0\n- Un **vecteur** est un tenseur de rang 1\n- Une **matrice** est un tenseur de rang 2\n- Un **tenseur 3D** est un tenseur de rang 3 (ex: image RGB)`,
      },
      {
        type: 'equation',
        content: '\\mathbf{T} \\in \\mathbb{R}^{d_1 \\times d_2 \\times \\cdots \\times d_n}',
        label: 'Tensor Shape',
      },
      {
        type: 'callout',
        content: '💡 En PyTorch, `torch.tensor()` crée un tenseur à partir de données Python. Utilisez `.shape` pour inspecter ses dimensions.',
      },
      {
        type: 'text',
        content: `Les opérations sur les tenseurs sont au cœur du calcul neuronal. La multiplication matricielle est l'opération la plus importante :`,
      },
      {
        type: 'equation',
        content: '\\mathbf{C} = \\mathbf{A} \\times \\mathbf{B} \\quad \\text{où } C_{ij} = \\sum_k A_{ik} B_{kj}',
        label: 'Matrix Multiplication',
        highlightVar: 'result',
      },
    ],
    exercises: [
      {
        id: 'tensors-ex1',
        title: 'Créer et manipuler des tenseurs',
        instructions: 'Créez un tenseur 3×4 rempli de uns, puis multipliez-le par un tenseur 4×2. Affichez la forme du résultat.',
        starterCode: `import torch

# Créez un tenseur 3x4 rempli de uns
A = ___

# Créez un tenseur 4x2 aléatoire
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

# Créez un tenseur 4x2 aléatoire
B = torch.randn(4, 2)

# Multiplication matricielle
result = torch.matmul(A, B)

print(f"Shape de A: {A.shape}")
print(f"Shape de B: {B.shape}")
print(f"Shape du résultat: {result.shape}")
print(f"Résultat:\\n{result}")`,
        hints: [
          'Utilisez torch.ones(rows, cols) pour créer un tenseur de uns',
          'torch.randn(rows, cols) génère des valeurs aléatoires normales',
          'torch.matmul(A, B) ou A @ B effectue la multiplication matricielle',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch

# ── Exploration des Tenseurs ──

# Scalaire (rang 0)
scalar = torch.tensor(42.0)
print(f"Scalaire: {scalar}, shape: {scalar.shape}")

# Vecteur (rang 1)
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vecteur: {vector}, shape: {vector.shape}")

# Matrice (rang 2)
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"Matrice shape: {matrix.shape}")

# Tenseur 3D (rang 3) — comme une image RGB
image = torch.randn(3, 224, 224)
print(f"Image tensor shape: {image.shape}")

# ── Opérations ──
A = torch.ones(3, 4)
B = torch.randn(4, 2)
result = A @ B
print(f"\\nMultiplication: {A.shape} × {B.shape} = {result.shape}")
`,
  },
  {
    id: 'perceptron',
    title: 'Le Perceptron & Neurone Artificiel',
    shortTitle: 'Perceptron',
    description: 'Comprendre le neurone artificiel, les poids, biais et la fonction d\'activation.',
    status: 'available',
    progress: 0,
    dependencies: ['tensors'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Le **perceptron** est le bloc élémentaire d'un réseau neuronal. Il prend des entrées, les multiplie par des **poids** (weights), ajoute un **biais** (bias), puis passe le résultat dans une **fonction d\'activation**.`,
      },
      {
        type: 'equation',
        content: 'y = \\sigma(\\mathbf{w}^T \\mathbf{x} + b)',
        label: 'Perceptron Output',
        highlightVar: 'output',
      },
      {
        type: 'text',
        content: `Où :\n- **w** : vecteur des poids\n- **x** : vecteur d\'entrée\n- **b** : biais\n- **σ** : fonction d\'activation (Sigmoid, ReLU, etc.)`,
      },
      {
        type: 'equation',
        content: '\\sigma(z) = \\frac{1}{1 + e^{-z}}',
        label: 'Sigmoid Function',
        highlightVar: 'sigmoid',
      },
      {
        type: 'callout',
        content: '🧠 Le biais permet au neurone de décaler sa frontière de décision, même quand toutes les entrées sont nulles.',
      },
    ],
    exercises: [
      {
        id: 'perceptron-ex1',
        title: 'Implémenter un neurone depuis zéro',
        instructions: 'Implémentez un neurone avec sigmoid qui prend 3 entrées.',
        starterCode: `import torch

def sigmoid(z):
    return ___

# Entrées
x = torch.tensor([1.0, 2.0, 3.0])

# Poids et biais (initialisés aléatoirement)
w = torch.randn(3)
b = torch.randn(1)

# Calcul du neurone
z = ___
output = ___

print(f"z = {z.item():.4f}")
print(f"output = {output.item():.4f}")`,
        solution: `import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# Entrées
x = torch.tensor([1.0, 2.0, 3.0])

# Poids et biais
w = torch.randn(3)
b = torch.randn(1)

# Calcul du neurone
z = torch.dot(w, x) + b
output = sigmoid(z)

print(f"z = {z.item():.4f}")
print(f"output = {output.item():.4f}")`,
        hints: [
          'sigmoid(z) = 1 / (1 + exp(-z))',
          'torch.dot() calcule le produit scalaire',
        ],
        completed: false,
      },
    ],
    codeTemplate: `import torch
import torch.nn as nn

# ── Le Neurone Artificiel ──

def sigmoid(z):
    """Fonction d'activation Sigmoid"""
    return 1 / (1 + torch.exp(-z))

# Données d'entrée
x = torch.tensor([1.0, 2.0, 3.0])

# Poids et biais
w = torch.randn(3)
b = torch.randn(1)

# Forward pass
z = torch.dot(w, x) + b
output = sigmoid(z)

print(f"Entrées  : {x}")
print(f"Poids    : {w}")
print(f"Biais    : {b.item():.4f}")
print(f"z = w·x + b = {z.item():.4f}")
print(f"σ(z) = {output.item():.4f}")
`,
  },
  {
    id: 'dense-network',
    title: 'Réseau Dense (Fully Connected)',
    shortTitle: 'Dense',
    description: 'Construire un réseau multicouche avec PyTorch nn.Module.',
    status: 'locked',
    progress: 0,
    dependencies: ['perceptron'],
    category: 'fundamentals',
    theory: [
      {
        type: 'text',
        content: `Un **réseau dense** (Fully Connected Network) empile plusieurs couches de neurones. Chaque neurone d\'une couche est connecté à tous les neurones de la couche suivante.`,
      },
      {
        type: 'equation',
        content: '\\mathbf{h}^{(l)} = \\sigma(\\mathbf{W}^{(l)} \\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)})',
        label: 'Layer Forward',
        highlightVar: 'hidden',
      },
      {
        type: 'callout',
        content: '⚡ En PyTorch, `nn.Linear(in_features, out_features)` crée une couche dense avec poids et biais automatiquement initialisés.',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

model = SimpleNet()
print(model)

# Compter les paramètres
total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal paramètres: {total:,}")
`,
  },
  {
    id: 'loss-functions',
    title: 'Fonctions de Perte (Loss)',
    shortTitle: 'Loss',
    description: 'Comprendre MSE, Cross-Entropy et comment mesurer l\'erreur.',
    status: 'locked',
    progress: 0,
    dependencies: ['dense-network'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `La **fonction de perte** mesure à quel point les prédictions du modèle sont éloignées de la réalité. C'est le signal qui guide l'apprentissage.`,
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2',
        label: 'Mean Squared Error',
        highlightVar: 'loss',
      },
      {
        type: 'equation',
        content: '\\mathcal{L}_{CE} = -\\sum_{i=1}^{C} y_i \\log(\\hat{y}_i)',
        label: 'Cross-Entropy Loss',
        highlightVar: 'loss',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ── Fonctions de Perte ──

# MSE pour la régression
predictions = torch.tensor([2.5, 3.2, 4.1])
targets = torch.tensor([3.0, 3.0, 4.0])

mse = nn.MSELoss()
loss = mse(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")

# Cross-Entropy pour la classification
logits = torch.tensor([[2.0, 1.0, 0.1]])
label = torch.tensor([0])

ce = nn.CrossEntropyLoss()
loss = ce(logits, label)
print(f"CE Loss: {loss.item():.4f}")
`,
  },
  {
    id: 'backprop',
    title: 'Backpropagation & Gradients',
    shortTitle: 'Backprop',
    description: 'Le cœur de l\'apprentissage : propagation arrière du gradient.',
    status: 'locked',
    progress: 0,
    dependencies: ['loss-functions'],
    category: 'training',
    theory: [
      {
        type: 'text',
        content: `La **backpropagation** utilise la règle de la chaîne pour calculer le gradient de la perte par rapport à chaque poids du réseau.`,
      },
      {
        type: 'equation',
        content: '\\frac{\\partial \\mathcal{L}}{\\partial w_{ij}} = \\frac{\\partial \\mathcal{L}}{\\partial a_j} \\cdot \\frac{\\partial a_j}{\\partial z_j} \\cdot \\frac{\\partial z_j}{\\partial w_{ij}}',
        label: 'Chain Rule',
        highlightVar: 'grad',
      },
    ],
    exercises: [],
    codeTemplate: `import torch

# ── Autograd : Backpropagation automatique ──

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward
y = w * x + b
loss = (y - 10) ** 2

print(f"y = {y.item():.2f}")
print(f"loss = {loss.item():.2f}")

# Backward
loss.backward()

print(f"\\n∂loss/∂w = {w.grad.item():.2f}")
print(f"∂loss/∂x = {x.grad.item():.2f}")
print(f"∂loss/∂b = {b.grad.item():.2f}")
`,
  },
  {
    id: 'optimizers',
    title: 'Optimiseurs (SGD, Adam)',
    shortTitle: 'Optimizers',
    description: 'Comment les poids sont mis à jour avec SGD, Momentum et Adam.',
    status: 'locked',
    progress: 0,
    dependencies: ['backprop'],
    category: 'training',
    theory: [
      {
        type: 'equation',
        content: 'w_{t+1} = w_t - \\eta \\cdot \\nabla_w \\mathcal{L}',
        label: 'SGD Update Rule',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    x = torch.randn(32, 1)
    y = 3 * x + 2 + torch.randn(32, 1) * 0.1
    
    pred = model(x)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

w, b = model.weight.item(), model.bias.item()
print(f"\\nAppris: y = {w:.2f}x + {b:.2f}")
print(f"Réel:   y = 3.00x + 2.00")
`,
  },
  {
    id: 'cnn',
    title: 'Réseaux Convolutifs (CNN)',
    shortTitle: 'CNN',
    description: 'Convolutions 2D, pooling et architectures pour la vision.',
    status: 'locked',
    progress: 0,
    dependencies: ['optimizers'],
    category: 'architectures',
    theory: [
      {
        type: 'equation',
        content: '(f * g)(i,j) = \\sum_m \\sum_n f(m,n) \\cdot g(i-m, j-n)',
        label: '2D Convolution',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
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
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(model)
`,
  },
  {
    id: 'rnn',
    title: 'Réseaux Récurrents (RNN/LSTM)',
    shortTitle: 'RNN',
    description: 'Traitement des séquences avec mémoire temporelle.',
    status: 'locked',
    progress: 0,
    dependencies: ['optimizers'],
    category: 'architectures',
    theory: [
      {
        type: 'equation',
        content: 'h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)',
        label: 'RNN Hidden State',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn

# ── RNN simple ──
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(1, 5, 10)  # batch=1, seq_len=5, features=10
output, h_n = rnn(x)
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {h_n.shape}")

# ── LSTM ──
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)
print(f"\\nLSTM Output: {output.shape}")
print(f"LSTM Hidden: {h_n.shape}")
print(f"LSTM Cell: {c_n.shape}")
`,
  },
  {
    id: 'attention',
    title: 'Mécanisme d\'Attention & Transformers',
    shortTitle: 'Attention',
    description: 'Self-Attention, Multi-Head Attention et l\'architecture Transformer.',
    status: 'locked',
    progress: 0,
    dependencies: ['rnn'],
    category: 'advanced',
    theory: [
      {
        type: 'equation',
        content: '\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
        label: 'Scaled Dot-Product Attention',
      },
    ],
    exercises: [],
    codeTemplate: `import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

attn = SelfAttention(d_model=64, n_heads=8)
x = torch.randn(1, 10, 64)
out = attn(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
`,
  },
];

// ── Graph positions for Roadmap visualization ──
export const nodePositions: Record<string, { x: number; y: number }> = {
  'tensors': { x: 150, y: 100 },
  'perceptron': { x: 400, y: 100 },
  'dense-network': { x: 650, y: 100 },
  'loss-functions': { x: 650, y: 280 },
  'backprop': { x: 400, y: 280 },
  'optimizers': { x: 150, y: 280 },
  'cnn': { x: 150, y: 460 },
  'rnn': { x: 400, y: 460 },
  'attention': { x: 650, y: 460 },
};

export function getTotalProgress(nodes: CourseNode[]): number {
  const completed = nodes.filter(n => n.status === 'completed').length;
  return Math.round((completed / nodes.length) * 100);
}
