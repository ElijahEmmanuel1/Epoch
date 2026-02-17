import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { Card } from "@/components/ui/Card";
import { BookOpen, Clock, Flame, ArrowRight, CheckCircle, Circle, Lock } from "lucide-react";

export const metadata: Metadata = {
    title: "Dashboard",
};

const chapters = [
    { id: 1, title: "Introduction", desc: "What is deep learning and why does it work?", status: "completed" as const, progress: 100 },
    { id: 2, title: "Linear Regression", desc: "Fitting functions to data from scratch.", status: "completed" as const, progress: 100 },
    { id: 3, title: "Shallow Networks", desc: "Universal approximation and hidden layers.", status: "in-progress" as const, progress: 45 },
    { id: 4, title: "Deep Networks", desc: "Composing representations for power.", status: "locked" as const, progress: 0 },
    { id: 5, title: "Loss Functions", desc: "Measuring and minimizing model error.", status: "locked" as const, progress: 0 },
    { id: 6, title: "Gradient Descent", desc: "Navigating the loss landscape.", status: "locked" as const, progress: 0 },
    { id: 7, title: "Backpropagation", desc: "Computing gradients efficiently.", status: "locked" as const, progress: 0 },
    { id: 8, title: "Regularization", desc: "Preventing overfitting in practice.", status: "locked" as const, progress: 0 },
    { id: 9, title: "Convolutional Networks", desc: "Learning spatial hierarchies.", status: "locked" as const, progress: 0 },
    { id: 10, title: "Transformers", desc: "Attention is all you need.", status: "locked" as const, progress: 0 },
];

const statusConfig = {
    completed: { icon: CheckCircle, badge: "success" as const, label: "Completed" },
    "in-progress": { icon: Circle, badge: "info" as const, label: "In Progress" },
    locked: { icon: Lock, badge: "default" as const, label: "Locked" },
};

export default function DashboardPage() {
    const completedCount = chapters.filter((c) => c.status === "completed").length;
    const overallProgress = Math.round((completedCount / chapters.length) * 100);
    const currentChapter = chapters.find((c) => c.status === "in-progress");

    return (
        <div className="max-w-5xl mx-auto space-y-10">

            {/* Header */}
            <div className="space-y-1">
                <h1 className="text-3xl font-bold tracking-tight text-foreground">Dashboard</h1>
                <p className="text-muted-foreground font-serif">
                    Pick up where you left off or explore a new chapter.
                </p>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Card className="p-6 space-y-3">
                    <div className="flex items-center gap-2 text-muted-foreground">
                        <BookOpen className="h-4 w-4" />
                        <span className="text-xs font-medium uppercase tracking-wider">Progress</span>
                    </div>
                    <div className="text-2xl font-bold text-foreground">{completedCount}/{chapters.length}</div>
                    <Progress value={overallProgress} variant="gradient" size="sm" showLabel />
                </Card>

                <Card className="p-6 space-y-3">
                    <div className="flex items-center gap-2 text-muted-foreground">
                        <Flame className="h-4 w-4" />
                        <span className="text-xs font-medium uppercase tracking-wider">Streak</span>
                    </div>
                    <div className="text-2xl font-bold text-foreground">5 days</div>
                    <p className="text-xs text-muted-foreground">Keep going! You&apos;re on fire.</p>
                </Card>

                <Card className="p-6 space-y-3">
                    <div className="flex items-center gap-2 text-muted-foreground">
                        <Clock className="h-4 w-4" />
                        <span className="text-xs font-medium uppercase tracking-wider">Study Time</span>
                    </div>
                    <div className="text-2xl font-bold text-foreground">12.5h</div>
                    <p className="text-xs text-muted-foreground">Total learning time this month.</p>
                </Card>
            </div>

            {/* Continue Learning Banner */}
            {currentChapter && (
                <Link href={`/chapter/${currentChapter.id}`} className="block group">
                    <Card variant="bordered" className="p-6 hover:border-foreground/20 transition-colors">
                        <div className="flex items-center justify-between gap-4">
                            <div className="flex-1 min-w-0">
                                <Badge variant="info" size="sm" className="mb-3">Continue Learning</Badge>
                                <h2 className="text-lg font-semibold text-foreground mb-1">
                                    Chapter {currentChapter.id}: {currentChapter.title}
                                </h2>
                                <p className="text-sm text-muted-foreground font-serif">{currentChapter.desc}</p>
                                <Progress value={currentChapter.progress} variant="gradient" size="sm" showLabel className="mt-4 max-w-xs" />
                            </div>
                            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all shrink-0" />
                        </div>
                    </Card>
                </Link>
            )}

            {/* All Chapters */}
            <div>
                <h2 className="text-lg font-semibold text-foreground mb-4">All Chapters</h2>
                <div className="grid gap-3 md:grid-cols-2">
                    {chapters.map((chapter) => {
                        const config = statusConfig[chapter.status];
                        const StatusIcon = config.icon;
                        const isAccessible = chapter.status !== "locked";

                        const cardContent = (
                            <Card
                                key={chapter.id}
                                className={`p-5 transition-all ${isAccessible
                                    ? "hover:border-foreground/20 hover:shadow-lg hover:shadow-black/5 cursor-pointer"
                                    : "opacity-60"
                                    }`}
                            >
                                <div className="flex items-start gap-4">
                                    <div className={`mt-0.5 shrink-0 ${chapter.status === "completed" ? "text-emerald-400" : chapter.status === "in-progress" ? "text-blue-400" : "text-muted-foreground/30"}`}>
                                        <StatusIcon className="h-5 w-5" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-[10px] font-mono text-muted-foreground tabular-nums">
                                                {String(chapter.id).padStart(2, "0")}
                                            </span>
                                            <Badge variant={config.badge} size="sm">{config.label}</Badge>
                                        </div>
                                        <h3 className="font-semibold text-foreground leading-snug">{chapter.title}</h3>
                                        <p className="text-sm text-muted-foreground font-serif mt-1">{chapter.desc}</p>
                                        {chapter.progress > 0 && chapter.progress < 100 && (
                                            <Progress value={chapter.progress} size="sm" variant="gradient" className="mt-3" />
                                        )}
                                    </div>
                                </div>
                            </Card>
                        );

                        if (isAccessible) {
                            return (
                                <Link key={chapter.id} href={`/chapter/${chapter.id}`}>
                                    {cardContent}
                                </Link>
                            );
                        }
                        return <div key={chapter.id}>{cardContent}</div>;
                    })}
                </div>
            </div>
        </div>
    );
}
