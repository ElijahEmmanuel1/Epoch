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
import { chaptersData } from "@/lib/data/chapters";
import { ShallowNetViz } from "@/components/chapter/3/ShallowNetViz";

// Map of interactive components
const InteractiveComponents: Record<string, React.ComponentType<any>> = {
    "ShallowNetViz": ShallowNetViz,
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

    if (!chapter || isNaN(chapterNum)) {
        notFound();
    }

    const prevChapter = chapterNum > 1 ? chapterNum - 1 : null;
    const nextChapter = chapterNum < totalChapters ? chapterNum + 1 : null;

    return (
        <div className="max-w-3xl mx-auto space-y-8 pb-32">
            {/* Breadcrumb */}
            <Breadcrumb
                items={[
                    { label: "Dashboard", href: "/dashboard" },
                    { label: `Chapter ${chapter.id}` },
                    { label: chapter.title },
                ]}
            />

            {/* Header */}
            <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                    <Badge variant="outline" size="md">Chapter {String(chapter.id).padStart(2, "0")}</Badge>
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                        <Clock className="h-3.5 w-3.5" />
                        <span className="text-xs">{chapter.estimatedTime}</span>
                    </div>
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
                    {chapter.title}
                </h1>
                <p className="text-lg text-muted-foreground font-serif leading-relaxed">
                    {chapter.description}
                </p>
            </div>

            {/* Progress */}
            <Progress value={chapterNum === 3 ? 15 : 45} variant="gradient" size="sm" showLabel />

            {/* Learning Objectives */}
            <Card variant="bordered" className="p-6">
                <div className="flex items-center gap-2 mb-4">
                    <BookOpen className="h-4 w-4 text-muted-foreground" />
                    <h2 className="font-semibold text-foreground text-sm">Learning Objectives</h2>
                </div>
                <ul className="space-y-2">
                    {chapter.objectives.map((obj, i) => (
                        <li key={i} className="flex items-start gap-3 text-sm text-muted-foreground font-serif">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            {obj}
                        </li>
                    ))}
                </ul>
            </Card>

            {/* Content */}
            <article className="prose-epoch space-y-6">
                {chapter.content.map((block, i) => {
                    switch (block.type) {
                        case "paragraph":
                            return (
                                <p key={i} className="text-muted-foreground font-serif leading-relaxed">
                                    {block.text}
                                </p>
                            );
                        case "heading":
                            return (
                                <h2 key={i} className="text-xl font-bold text-foreground mt-12 mb-4 tracking-tight border-b border-white/5 pb-2">
                                    {block.text}
                                </h2>
                            );
                        case "math-block":
                            return (
                                <div key={i} className="py-6 flex justify-center rounded-xl bg-muted/30 border border-border my-6 overflow-x-auto">
                                    <Math latex={block.latex!} block />
                                </div>
                            );
                        case "interactive": {
                            const Component = block.component ? InteractiveComponents[block.component] : null;
                            if (!Component) return <div key={i} className="text-red-500">Missing Component: {block.component}</div>;

                            return (
                                <div key={i} className="my-10">
                                    <Component {...block.props} />
                                </div>
                            );
                        }
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
                            const variant = block.variant || "info";
                            const CalloutIcon = icons[variant];
                            return (
                                <div key={i} className={`p-5 rounded-xl border ${colors[variant]} my-6`}>
                                    <div className="flex items-center gap-2 mb-2">
                                        <CalloutIcon className={`h-4 w-4 ${iconColors[variant]}`} />
                                        <h3 className="font-semibold text-foreground text-sm">{block.title || "Note"}</h3>
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
            <div className="flex items-center justify-between pt-8 border-t border-border mt-16">
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
