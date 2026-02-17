import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { ArrowRight, BookOpen, Terminal, Brain, Sparkles, Github, Cpu, Layers } from "lucide-react";
import { Animated, FloatingGlow } from "@/components/landing/Animations";
import { Badge } from "@/components/ui/Badge";

const features = [
    {
        icon: Terminal,
        title: "Code-First Learning",
        desc: "Don't just read equations — implement them. Interactive, runnable code environments for every concept covered.",
    },
    {
        icon: BookOpen,
        title: "Rigorous Theory",
        desc: "Full mathematical derivations rendered beautifully. Build deep intuition alongside formal proofs.",
    },
    {
        icon: Cpu,
        title: "Production Patterns",
        desc: "Learn using modern stacks: from raw NumPy fundamentals to PyTorch and industrial-grade architectures.",
    },
];

const chapters = [
    { num: "01", title: "Introduction", desc: "What is deep learning and why does it work?" },
    { num: "02", title: "Linear Regression", desc: "Fitting functions to data from scratch." },
    { num: "03", title: "Shallow Networks", desc: "Universal approximation and hidden layers." },
    { num: "04", title: "Deep Networks", desc: "Composing representations for power." },
    { num: "05", title: "Loss Functions", desc: "Measuring and minimizing model error." },
    { num: "06", title: "Gradient Descent", desc: "Navigating the loss landscape." },
];

const stats = [
    { value: "10+", label: "Chapters" },
    { value: "50+", label: "Interactive Demos" },
    { value: "100%", label: "Open Source" },
];

export default function Home() {
    return (
        <div className="flex flex-col min-h-screen bg-background text-foreground font-sans">

            {/* ═══════════════════ HEADER ═══════════════════ */}
            <header className="fixed top-0 w-full z-50 border-b border-border bg-background/70 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2.5 group">
                        <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-foreground to-foreground/70 flex items-center justify-center shadow-sm">
                            <span className="text-background font-bold text-xs">E</span>
                        </div>
                        <span className="font-bold text-sm tracking-tight text-foreground">Epoch</span>
                    </Link>

                    <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-muted-foreground" aria-label="Main navigation">
                        <a href="#features" className="hover:text-foreground transition-colors">Features</a>
                        <a href="#syllabus" className="hover:text-foreground transition-colors">Syllabus</a>
                        <a href="https://github.com/ElijahEmmanuel1/Epoch" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors inline-flex items-center gap-1.5">
                            <Github className="h-3.5 w-3.5" />
                            GitHub
                        </a>
                    </nav>

                    <div className="flex items-center gap-3">
                        <Link href="/dashboard">
                            <Button variant="ghost" size="sm" className="hidden sm:inline-flex">Sign In</Button>
                        </Link>
                        <Link href="/dashboard">
                            <Button variant="primary" size="sm">Get Started</Button>
                        </Link>
                    </div>
                </div>
            </header>

            {/* ═══════════════════ HERO ═══════════════════ */}
            <main className="flex-1">
                <section className="relative overflow-hidden pt-32 pb-24 md:pt-44 md:pb-32">
                    {/* Background effects */}
                    <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
                        <FloatingGlow className="absolute top-[-15%] left-[-10%] w-[50%] h-[50%] bg-blue-500/[0.07] blur-[120px] rounded-full" />
                        <FloatingGlow className="absolute bottom-[-15%] right-[-10%] w-[50%] h-[50%] bg-violet-500/[0.07] blur-[120px] rounded-full" />
                        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[500px] bg-gradient-to-b from-muted/20 to-transparent" />
                    </div>

                    <div className="max-w-5xl mx-auto px-6 text-center relative z-10">
                        <Animated delay={0}>
                            <Badge variant="info" size="md" className="mb-8">
                                <span className="flex h-1.5 w-1.5 rounded-full bg-blue-400 mr-2 animate-pulse" />
                                v1.0 Public Beta
                            </Badge>
                        </Animated>

                        <Animated delay={1}>
                            <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold tracking-tight mb-8 text-gradient leading-[1.1] pb-2">
                                Master Deep Learning{" "}
                                <br className="hidden sm:block" />
                                <span className="text-muted-foreground">From First Principles.</span>
                            </h1>
                        </Animated>

                        <Animated delay={2}>
                            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-12 font-serif leading-relaxed">
                                An interactive, code-first exploration of modern AI. Based on the
                                landmark text &quot;Understanding Deep Learning&quot; by Simon J.D. Prince.
                            </p>
                        </Animated>

                        <Animated delay={3}>
                            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                                <Link href="/dashboard">
                                    <Button size="lg" className="min-w-[200px] group">
                                        Start Learning
                                        <ArrowRight className="w-4 h-4 ml-2 opacity-50 group-hover:translate-x-1 transition-transform" />
                                    </Button>
                                </Link>
                                <Link href="#syllabus">
                                    <Button variant="secondary" size="lg" className="min-w-[200px]">
                                        <BookOpen className="w-4 h-4 mr-2 opacity-50" />
                                        View Syllabus
                                    </Button>
                                </Link>
                            </div>
                        </Animated>

                        {/* Stats */}
                        <Animated delay={5}>
                            <div className="flex items-center justify-center gap-8 md:gap-16 mt-16 pt-8 border-t border-border">
                                {stats.map((stat) => (
                                    <div key={stat.label} className="text-center">
                                        <div className="text-2xl md:text-3xl font-bold text-foreground">{stat.value}</div>
                                        <div className="text-xs text-muted-foreground mt-1">{stat.label}</div>
                                    </div>
                                ))}
                            </div>
                        </Animated>
                    </div>
                </section>

                {/* ═══════════════════ FEATURES ═══════════════════ */}
                <section id="features" className="py-24 md:py-32 border-t border-border">
                    <div className="max-w-6xl mx-auto px-6">
                        <Animated>
                            <div className="text-center mb-16">
                                <Badge variant="outline" size="md" className="mb-4">
                                    <Sparkles className="h-3 w-3 mr-1.5" />
                                    Features
                                </Badge>
                                <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground mb-4">
                                    Built for serious learners
                                </h2>
                                <p className="text-muted-foreground font-serif max-w-xl mx-auto">
                                    Every design decision optimized for deep understanding — not passive consumption.
                                </p>
                            </div>
                        </Animated>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {features.map((feature, i) => (
                                <Animated key={i} delay={i + 1} variant="scaleIn">
                                    <div className="group p-8 rounded-2xl border border-border bg-card/50 hover:bg-card hover:shadow-lg hover:shadow-black/5 transition-all duration-300 h-full">
                                        <div className="h-12 w-12 rounded-xl bg-muted border border-border flex items-center justify-center mb-6 text-muted-foreground group-hover:text-foreground group-hover:border-foreground/20 transition-colors">
                                            <feature.icon className="w-5 h-5" />
                                        </div>
                                        <h3 className="text-foreground font-semibold text-lg mb-3">{feature.title}</h3>
                                        <p className="text-sm text-muted-foreground font-serif leading-relaxed">{feature.desc}</p>
                                    </div>
                                </Animated>
                            ))}
                        </div>
                    </div>
                </section>

                {/* ═══════════════════ SYLLABUS ═══════════════════ */}
                <section id="syllabus" className="py-24 md:py-32 border-t border-border bg-card/30">
                    <div className="max-w-4xl mx-auto px-6">
                        <Animated>
                            <div className="text-center mb-16">
                                <Badge variant="outline" size="md" className="mb-4">
                                    <Layers className="h-3 w-3 mr-1.5" />
                                    Curriculum
                                </Badge>
                                <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground mb-4">
                                    A structured path to mastery
                                </h2>
                                <p className="text-muted-foreground font-serif max-w-xl mx-auto">
                                    10 carefully sequenced chapters covering the foundations and frontiers of deep learning.
                                </p>
                            </div>
                        </Animated>

                        <div className="space-y-3">
                            {chapters.map((ch, i) => (
                                <Animated key={ch.num} delay={i * 0.5}>
                                    <Link href={`/chapter/${parseInt(ch.num)}`}>
                                        <div className="group flex items-center gap-6 p-5 rounded-xl border border-border bg-card/50 hover:bg-card hover:border-foreground/10 hover:shadow-lg hover:shadow-black/5 transition-all duration-300">
                                            <span className="text-sm font-mono text-muted-foreground/50 group-hover:text-muted-foreground transition-colors tabular-nums w-8">
                                                {ch.num}
                                            </span>
                                            <div className="flex-1 min-w-0">
                                                <h3 className="font-semibold text-foreground group-hover:text-foreground transition-colors">
                                                    {ch.title}
                                                </h3>
                                                <p className="text-sm text-muted-foreground font-serif mt-0.5 truncate">
                                                    {ch.desc}
                                                </p>
                                            </div>
                                            <ArrowRight className="h-4 w-4 text-muted-foreground/30 group-hover:text-foreground group-hover:translate-x-1 transition-all shrink-0" />
                                        </div>
                                    </Link>
                                </Animated>
                            ))}
                        </div>

                        <Animated delay={4}>
                            <div className="text-center mt-10">
                                <Link href="/dashboard">
                                    <Button variant="secondary" size="lg">
                                        View All Chapters
                                        <ArrowRight className="w-4 h-4 ml-2" />
                                    </Button>
                                </Link>
                            </div>
                        </Animated>
                    </div>
                </section>

                {/* ═══════════════════ CTA ═══════════════════ */}
                <section className="py-24 md:py-32 border-t border-border">
                    <div className="max-w-3xl mx-auto px-6 text-center">
                        <Animated>
                            <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-gradient-to-br from-foreground to-foreground/70 mb-8 shadow-lg">
                                <Brain className="h-7 w-7 text-background" />
                            </div>
                        </Animated>
                        <Animated delay={1}>
                            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground mb-4">
                                Ready to begin?
                            </h2>
                        </Animated>
                        <Animated delay={2}>
                            <p className="text-muted-foreground font-serif max-w-lg mx-auto mb-8">
                                Join thousands of learners building a rigorous understanding of deep learning from the ground up.
                            </p>
                        </Animated>
                        <Animated delay={3}>
                            <Link href="/dashboard">
                                <Button size="lg" className="min-w-[220px] group">
                                    Start Learning — Free
                                    <ArrowRight className="w-4 h-4 ml-2 opacity-50 group-hover:translate-x-1 transition-transform" />
                                </Button>
                            </Link>
                        </Animated>
                    </div>
                </section>
            </main>

            {/* ═══════════════════ FOOTER ═══════════════════ */}
            <footer className="border-t border-border py-16 bg-card/30" role="contentinfo">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
                        {/* Brand */}
                        <div className="md:col-span-1 space-y-4">
                            <Link href="/" className="flex items-center gap-2.5">
                                <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-foreground to-foreground/70 flex items-center justify-center">
                                    <span className="text-background font-bold text-xs">E</span>
                                </div>
                                <span className="font-bold text-sm tracking-tight text-foreground">Epoch</span>
                            </Link>
                            <p className="text-sm text-muted-foreground font-serif leading-relaxed">
                                An open-source interactive platform for mastering deep learning.
                            </p>
                        </div>

                        {/* Links */}
                        <div>
                            <h4 className="text-sm font-semibold text-foreground mb-4">Platform</h4>
                            <ul className="space-y-3 text-sm text-muted-foreground">
                                <li><Link href="/dashboard" className="hover:text-foreground transition-colors">Dashboard</Link></li>
                                <li><Link href="/chapter/1" className="hover:text-foreground transition-colors">Start Learning</Link></li>
                                <li><a href="#syllabus" className="hover:text-foreground transition-colors">Syllabus</a></li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="text-sm font-semibold text-foreground mb-4">Resources</h4>
                            <ul className="space-y-3 text-sm text-muted-foreground">
                                <li><a href="https://udlbook.github.io/udlbook/" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">Original Textbook</a></li>
                                <li><a href="https://github.com/ElijahEmmanuel1/Epoch" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">GitHub</a></li>
                                <li><a href="https://github.com/ElijahEmmanuel1/Epoch/issues" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">Report a Bug</a></li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="text-sm font-semibold text-foreground mb-4">Legal</h4>
                            <ul className="space-y-3 text-sm text-muted-foreground">
                                <li><a href="/privacy" className="hover:text-foreground transition-colors">Privacy Policy</a></li>
                                <li><a href="/terms" className="hover:text-foreground transition-colors">Terms of Service</a></li>
                            </ul>
                        </div>
                    </div>

                    <div className="pt-8 border-t border-border flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-muted-foreground">
                        <p>&copy; {new Date().getFullYear()} Epoch. Open source education for everyone.</p>
                        <p className="font-serif">
                            Based on &quot;Understanding Deep Learning&quot; by Simon J.D. Prince
                        </p>
                    </div>
                </div>
            </footer>
        </div>
    );
}
