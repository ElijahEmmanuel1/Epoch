import { Math } from "@/components/mathematics/Math";

export default async function ChapterPage({ params }: { params: Promise<{ chapterId: string }> }) {
    const { chapterId } = await params;

    return (
        <div className="max-w-3xl mx-auto space-y-8">
            <div className="space-y-2">
                <div className="text-sm font-medium text-muted-foreground uppercase tracking-widest">
                    Chapter {chapterId}
                </div>
                <h1 className="text-4xl font-bold tracking-tight">Introduction</h1>
            </div>

            <div className="prose prose-slate dark:prose-invert max-w-none">
                <p className="lead text-xl text-muted-foreground font-serif">
                    Deep learning is a subset of machine learning that is concerned with artificial neural networks.
                </p>

                <p>
                    We can describe a model mathematically as a function <Math latex="f(x; \phi)" /> that maps an input <Math latex="x" /> to an output <Math latex="y" />.
                </p>

                <div className="my-8 p-6 bg-muted/30 rounded-lg border">
                    <h3 className="font-sans font-semibold mb-2">Key Concept: The Model</h3>
                    <p>
                        The goal is to find parameters <Math latex="\phi" /> such that:
                    </p>
                    <div className="py-4 flex justify-center">
                        <Math latex="y \approx f(x; \phi)" block />
                    </div>
                </div>
            </div>
        </div>
    );
}
