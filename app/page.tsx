import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { ArrowRight, BookOpen, Terminal } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground font-sans selection:bg-zinc-800 selection:text-zinc-100">

      {/* Header */}
      <header className="fixed top-0 w-full z-50 border-b border-white/5 bg-background/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-6 w-6 rounded bg-zinc-100 flex items-center justify-center">
              <span className="text-zinc-900 font-bold text-xs">E</span>
            </div>
            <span className="font-bold text-sm tracking-tight text-zinc-100">Epoch</span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-zinc-400">
            <a href="#syllabus" className="hover:text-zinc-100 transition-colors">Syllabus</a>
            <a href="#about" className="hover:text-zinc-100 transition-colors">About</a>
            <a href="https://github.com/ElijahEmmanuel1/Epoch" target="_blank" className="hover:text-zinc-100 transition-colors">GitHub</a>
          </nav>
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="sm" className="hidden sm:inline-flex">Sign In</Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="primary" size="sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center relative overflow-hidden pt-32 pb-20">

        {/* Background Gradients */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[500px] bg-gradient-to-b from-zinc-800/20 to-transparent pointer-events-none" />
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/5 blur-[120px] rounded-full pointer-events-none" />

        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <div className="inline-flex items-center rounded-full border border-zinc-800 bg-zinc-900/50 px-3 py-1 text-xs font-medium text-zinc-400 backdrop-blur-md mb-8">
            <span className="flex h-2 w-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>
            v1.0 Public Beta
          </div>

          <h1 className="text-5xl sm:text-7xl font-bold tracking-tight mb-8 text-transparent bg-clip-text bg-gradient-to-b from-white to-white/50 pb-2">
            Master Deep Learning <br />
            <span className="text-zinc-500">From First Principles.</span>
          </h1>

          <p className="text-xl text-zinc-400 max-w-2xl mx-auto mb-12 font-serif leading-relaxed">
            An interactive, code-first exploration of modern AI. Based on the
            landmark text &quot;Understanding Deep Learning&quot; by Simon J.D. Prince.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/dashboard">
              <Button size="lg" className="min-w-[180px] group">
                Start Learning
                <ArrowRight className="w-4 h-4 ml-2 opacity-50 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="/syllabus">
              <Button variant="secondary" size="lg" className="min-w-[180px]">
                <BookOpen className="w-4 h-4 mr-2 opacity-50" />
                View Syllabus
              </Button>
            </Link>
          </div>

          {/* Feature Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 text-left">
            {[
              {
                icon: Terminal,
                title: "Code-First",
                desc: "Don't just read equations. Implement them. Interactive code environments for every concept."
              },
              {
                icon: BookOpen,
                title: "Rigorous Theory",
                desc: "Full mathematical derivations visualized dynamically. Build intuition alongside formal proofs."
              },
              {
                icon: ArrowRight,
                title: "Production Ready",
                desc: "Learn using modern stacks. From raw NumPy to PyTorch and industrial-grade architectures."
              }
            ].map((feature, i) => (
              <div key={i} className="p-6 rounded-2xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
                <div className="h-10 w-10 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center mb-4 text-zinc-400">
                  <feature.icon className="w-5 h-5" />
                </div>
                <h3 className="text-zinc-100 font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-zinc-500 font-serif leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      <footer className="border-t border-white/5 py-12 bg-background relative z-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6 text-sm text-zinc-500">
          <p>&copy; {new Date().getFullYear()} Epoch Platform. Open Source Education.</p>
          <div className="flex gap-6">
            <a href="#" className="hover:text-zinc-300 transition-colors">Privacy</a>
            <a href="#" className="hover:text-zinc-300 transition-colors">Terms</a>
            <a href="#" className="hover:text-zinc-300 transition-colors">Twitter</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
