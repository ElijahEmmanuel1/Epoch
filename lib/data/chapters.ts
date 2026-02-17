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
        component?: string; // e.g., "ShallowNetViz"
        props?: Record<string, any>; // Props for the interactive component
    }[];
}

export const chaptersData: Record<string, ChapterData> = {
    "1": {
        id: 1,
        title: "Introduction",
        description: "An overview of deep learning: what it is, why it matters, and how models learn from data.",
        estimatedTime: "25 min",
        objectives: [
            "Understand what deep learning is and its relationship to machine learning",
            "Define a model as a parameterized function",
            "Understand the concept of supervised learning",
        ],
        content: [
            { type: "paragraph", text: "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input." },
            { type: "heading", text: "What is a Model?" },
            { type: "paragraph", text: "We can describe a model mathematically as a function f(x; φ) that maps an input x to an output y. The semicolon separates the input data from the model parameters." },
            { type: "callout", variant: "tip", title: "Key Concept: The Model", text: "The goal of learning is to find parameters φ such that the model produces useful predictions:" },
            { type: "math-block", latex: "y \\approx f(x; \\phi)" },
            { type: "heading", text: "Supervised Learning" },
            { type: "paragraph", text: "In supervised learning, we are given a training dataset of input-output pairs {(xᵢ, yᵢ)}. The goal is to learn a function that generalizes well to unseen data — not simply memorize the training examples." },
            { type: "callout", variant: "info", title: "Loss Function", text: "To measure how well our model fits the data, we define a loss function L(φ) that quantifies the discrepancy between predictions and ground truth:" },
            { type: "math-block", latex: "L(\\phi) = \\sum_{i=1}^{N} \\ell\\big(f(x_i; \\phi),\\, y_i\\big)" },
            { type: "paragraph", text: "The training process consists of finding the parameters that minimize this loss. This optimization problem is the heart of all machine learning." },
        ],
    },
    // We will populate Chapter 3 fully in the next steps
    "3": {
        id: 3,
        title: "Shallow Neural Networks",
        description: "Universal approximation: how a single hidden layer can represent any continuous function.",
        estimatedTime: "45 min",
        objectives: [
            "Construct a neural network with one hidden layer",
            "Understand the role of activation functions (ReLU)",
            "Visualize how hidden units combine to form complex functions",
        ],
        content: [
            { type: "paragraph", text: "In the previous chapter, we saw how linear models could fit straight lines to data. However, most real-world relationships are non-linear. To model complex data, we need a more flexible family of functions." },
            { type: "heading", text: "The Neural Network Equation" },
            { type: "paragraph", text: "A shallow neural network with one hidden layer takes a scalar input x, passes it through D hidden units, and produces a scalar output y. The equation is:" },
            { type: "math-block", latex: "f(x; \\phi) = \\beta_0 + \\sum_{d=1}^{D} \\omega_d \\cdot a(\\beta_d + \\omega_{d0} x)" },
            { type: "paragraph", text: "Let's break this down piece by piece. Inside the activation function a(·), we have a linear function of the input:" },
            { type: "math-block", latex: "\\beta_d + \\omega_{d0} x" },
            { type: "paragraph", text: "This is passed through a non-linear activation function, usually ReLU:" },
            { type: "math-block", latex: "a(z) = \\text{ReLU}(z) = \\max(0, z)" },
            { type: "paragraph", text: "The output of each hidden unit is then weighted by ωd and summed up, with a final bias β0 added." },

            { type: "heading", text: "Visualizing Hidden Units" },
            { type: "paragraph", text: "Each hidden unit contributes a 'hinge' shape to the final function. By combining these hinges, we can approximate any continuous curve. Interact with the visualization below to see how changing parameters affects the shape." },

            { type: "interactive", component: "ShallowNetViz", props: { defaultUnits: 3 } },

            { type: "heading", text: "Universal Approximation Theorem" },
            { type: "paragraph", text: "One of the most profound results in deep learning is that a shallow network with enough hidden units can approximate any continuous function to arbitrary precision within a compact domain. This is known as the Universal Approximation Theorem." },
            { type: "callout", variant: "tip", title: "Intuition", text: "Think of each hidden unit as a building block. By adding more blocks (increasing width D), we can construct increasingly complex shapes, just like building with LEGO bricks." },
        ]
    }
};
