import { Math } from "@/components/mathematics/Math";
import { Breadcrumb } from "@/components/ui/Breadcrumb";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Progress } from "@/components/ui/Progress";
import Link from "next/link";
import type { Metadata } from "next";
import { ArrowLeft, ArrowRight, Clock, BookOpen, Lightbulb, Code } from "lucide-react";
import { notFound } from "next/navigation";

interface ChapterData {
    id: number;
    title: string;
    description: string;
    estimatedTime: string;
    objectives: string[];
    content: {
        type: "paragraph" | "heading" | "math-block" | "callout" | "code-note";
        text?: string;
        latex?: string;
        title?: string;
        variant?: "info" | "warning" | "tip";
    }[];
}

const chaptersData: Record<string, ChapterData> = {
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
    "2": {
        id: 2,
        title: "Linear Regression",
        description: "Fitting linear functions to data — the simplest model, and a foundation for everything that follows.",
        estimatedTime: "30 min",
        objectives: [
            "Derive the linear regression model from first principles",
            "Understand least-squares loss and its properties",
            "Compute the closed-form solution for linear regression",
        ],
        content: [
            { type: "paragraph", text: "Linear regression is the simplest parametric model. Despite its simplicity, it introduces all the core concepts of model fitting: parameters, loss functions, and optimization." },
            { type: "heading", text: "The Linear Model" },
            { type: "paragraph", text: "A linear model maps a scalar input x to a scalar output y using two parameters — a slope (ω) and an intercept (β):" },
            { type: "math-block", latex: "f(x; \\phi) = \\beta + \\omega x" },
            { type: "callout", variant: "tip", title: "Parameters", text: "Here φ = {β, ω} represents our model parameters. β controls the y-intercept, ω controls the slope." },
            { type: "heading", text: "Least Squares Loss" },
            { type: "paragraph", text: "We measure the quality of the fit using the sum of squared residuals between predictions and targets:" },
            { type: "math-block", latex: "L(\\phi) = \\sum_{i=1}^{N} \\big(y_i - f(x_i; \\phi)\\big)^2" },
            { type: "paragraph", text: "This convex loss function has a unique global minimum that can be found analytically by taking the derivative and setting it to zero." },
        ],
    },
    "3": {
        id: 3,
        title: "Shallow Neural Networks",
        description: "Universal approximation: how a single hidden layer can represent any continuous function.",
        estimatedTime: "35 min",
        objectives: [
            "Construct a neural network with one hidden layer",
            "Understand activation functions (ReLU)",
            "Grasp the universal approximation theorem intuitively",
        ],
        content: [
            { type: "paragraph", text: "A shallow neural network adds a hidden layer of nonlinear transformations between input and output. This simple addition gives the model dramatically more expressive power." },
            { type: "heading", text: "Architecture" },
            { type: "paragraph", text: "A shallow network with D hidden units computes:" },
            { type: "math-block", latex: "f(x; \\phi) = \\beta_0 + \\sum_{d=1}^{D} \\omega_d \\cdot a\\big(\\theta_{d0} + \\theta_{d1} x\\big)" },
            { type: "callout", variant: "info", title: "Activation Function", text: "The function a(·) is a nonlinear activation function. The most common choice today is the ReLU (Rectified Linear Unit):" },
            { type: "math-block", latex: "\\text{ReLU}(z) = \\max(0, z)" },
            { type: "paragraph", text: "Each hidden unit contributes a 'hinge' — a piecewise linear segment. By combining many hinges with different slopes and positions, the network can approximate any continuous function." },
        ],
    },
};

// Generate metadata dynamically
export async function generateMetadata({ params }: { params: Promise<{ chapterId: string }> }): Promise<Metadata> {
    const { chapterId } = await params;
    const chapter = chaptersData[chapterId];
    if (!chapter) return { title: "Chapter Not Found" };
    return {
        title: `Ch. ${chapter.id}: ${chapter.title}`,
        description: chapter.description,
    };
}

const totalChapters = 10;

export default async function ChapterPage({ params }: { params: Promise<{ chapterId: string }> }) {
    const { chapterId } = await params;
    const chapter = chaptersData[chapterId];
    const chapterNum = parseInt(chapterId);

    if (!chapter && (isNaN(chapterNum) || chapterNum < 1 || chapterNum > totalChapters)) {
        notFound();
    }

    // Fallback for chapters without detailed content yet
    const data: ChapterData = chapter || {
        id: chapterNum,
        title: `Chapter ${chapterNum}`,
        description: "This chapter is coming soon. Check back for updates.",
        estimatedTime: "—",
        objectives: ["Content coming soon"],
        content: [
            { type: "callout", variant: "info", title: "Coming Soon", text: "This chapter's interactive content is currently being developed. Stay tuned!" },
        ],
    };

    const prevChapter = chapterNum > 1 ? chapterNum - 1 : null;
    const nextChapter = chapterNum < totalChapters ? chapterNum + 1 : null;

    return (
        <div className="max-w-3xl mx-auto space-y-8">
            {/* Breadcrumb */}
            <Breadcrumb
                items={[
                    { label: "Dashboard", href: "/dashboard" },
                    { label: `Chapter ${data.id}` },
                    { label: data.title },
                ]}
            />

            {/* Header */}
            <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                    <Badge variant="outline" size="md">Chapter {String(data.id).padStart(2, "0")}</Badge>
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                        <Clock className="h-3.5 w-3.5" />
                        <span className="text-xs">{data.estimatedTime}</span>
                    </div>
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
                    {data.title}
                </h1>
                <p className="text-lg text-muted-foreground font-serif leading-relaxed">
                    {data.description}
                </p>
            </div>

            {/* Progress */}
            <Progress value={chapter ? 45 : 0} variant="gradient" size="sm" showLabel />

            {/* Learning Objectives */}
            <Card variant="bordered" className="p-6">
                <div className="flex items-center gap-2 mb-4">
                    <BookOpen className="h-4 w-4 text-muted-foreground" />
                    <h2 className="font-semibold text-foreground text-sm">Learning Objectives</h2>
                </div>
                <ul className="space-y-2">
                    {data.objectives.map((obj, i) => (
                        <li key={i} className="flex items-start gap-3 text-sm text-muted-foreground font-serif">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            {obj}
                        </li>
                    ))}
                </ul>
            </Card>

            {/* Content */}
            <article className="prose-epoch space-y-6">
                {data.content.map((block, i) => {
                    switch (block.type) {
                        case "paragraph":
                            return (
                                <p key={i} className="text-muted-foreground font-serif leading-relaxed">
                                    {block.text}
                                </p>
                            );
                        case "heading":
                            return (
                                <h2 key={i} className="text-xl font-bold text-foreground mt-10 mb-4 tracking-tight">
                                    {block.text}
                                </h2>
                            );
                        case "math-block":
                            return (
                                <div key={i} className="py-6 flex justify-center rounded-xl bg-muted/30 border border-border my-6">
                                    <Math latex={block.latex!} block />
                                </div>
                            );
                        case "callout": {
                            const icons = {
                                info: Code,
                                warning: Lightbulb,
                                tip: Lightbulb,
                            };
                            const colors = {
                                info: "border-blue-500/20 bg-blue-500/5",
                                warning: "border-amber-500/20 bg-amber-500/5",
                                tip: "border-emerald-500/20 bg-emerald-500/5",
                            };
                            const iconColors = {
                                info: "text-blue-400",
                                warning: "text-amber-400",
                                tip: "text-emerald-400",
                            };
                            const CalloutIcon = icons[block.variant || "info"];
                            return (
                                <div key={i} className={`p-5 rounded-xl border ${colors[block.variant || "info"]} my-6`}>
                                    <div className="flex items-center gap-2 mb-2">
                                        <CalloutIcon className={`h-4 w-4 ${iconColors[block.variant || "info"]}`} />
                                        <h3 className="font-semibold text-foreground text-sm">{block.title}</h3>
                                    </div>
                                    <p className="text-sm text-muted-foreground font-serif leading-relaxed">
                                        {block.text}
                                    </p>
                                </div>
                            );
                        }
                        default:
                            return null;
                    }
                })}
            </article>

            {/* Navigation */}
            <div className="flex items-center justify-between pt-8 border-t border-border">
                {prevChapter ? (
                    <Link href={`/chapter/${prevChapter}`}>
                        <Button variant="ghost" size="md">
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            Previous
                        </Button>
                    </Link>
                ) : (
                    <div />
                )}
                {nextChapter ? (
                    <Link href={`/chapter/${nextChapter}`}>
                        <Button variant="primary" size="md">
                            Next Chapter
                            <ArrowRight className="h-4 w-4 ml-2" />
                        </Button>
                    </Link>
                ) : (
                    <Link href="/dashboard">
                        <Button variant="primary" size="md">
                            Back to Dashboard
                            <ArrowRight className="h-4 w-4 ml-2" />
                        </Button>
                    </Link>
                )}
            </div>
        </div>
    );
}
