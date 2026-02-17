import Link from "next/link";
import { BookOpen, Home, Layers, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
    { name: "Overview", href: "/dashboard", icon: Home },
    { name: "The Fundamentals", href: "/chapter/1", icon: Layers },
    { name: "Training & Optimization", href: "/chapter/5", icon: Zap },
    { name: "Architectures", href: "/chapter/10", icon: BookOpen },
];

export function Sidebar({ className }: { className?: string }) {
    return (
        <div className={cn("glass border-r border-white/5 w-64 min-h-screen flex flex-col sticky top-0", className)}>
            <div className="p-6 border-b border-white/5">
                <Link href="/" className="flex items-center gap-2">
                    <div className="h-6 w-6 rounded bg-zinc-100 flex items-center justify-center">
                        <span className="text-zinc-900 font-bold text-xs">E</span>
                    </div>
                    <span className="font-bold text-sm tracking-tight text-zinc-100">Epoch</span>
                </Link>
            </div>

            <div className="flex-1 py-6 px-3 space-y-6 overflow-y-auto">

                {/* Main Navigation */}
                <div className="space-y-1">
                    {navigation.map((item) => (
                        <Link
                            key={item.name}
                            href={item.href}
                            className="flex items-center gap-3 px-3 py-2 text-sm font-medium text-zinc-400 rounded-lg hover:bg-white/5 hover:text-zinc-100 transition-all group"
                        >
                            <item.icon className="h-4 w-4 text-zinc-500 group-hover:text-zinc-100 transition-colors" />
                            {item.name}
                        </Link>
                    ))}
                </div>

                {/* Detailed Syllabus */}
                <div>
                    <h3 className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
                        Syllabus
                    </h3>
                    <div className="space-y-0.5">
                        <SidebarItem href="/chapter/1" number="01" title="Introduction" />
                        <SidebarItem href="/chapter/2" number="02" title="Linear Regression" />
                        <SidebarItem href="/chapter/3" number="03" title="Shallow Nets" />
                        <SidebarItem href="/chapter/4" number="04" title="Deep Nets" />
                    </div>
                </div>
            </div>

            {/* User / Footer */}
            <div className="p-4 border-t border-white/5">
                <div className="flex items-center gap-3 p-2 rounded-lg bg-white/5 border border-white/5">
                    <div className="h-8 w-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 text-xs font-bold">
                        US
                    </div>
                    <div className="flex-1 overflow-hidden">
                        <p className="text-xs font-medium text-zinc-200 truncate">User Account</p>
                        <p className="text-[10px] text-zinc-500 truncate">Pro Plan</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

function SidebarItem({ href, number, title }: { href: string; number: string; title: string }) {
    return (
        <Link
            href={href}
            className="flex items-center gap-3 px-3 py-2 text-sm text-zinc-500 rounded-lg hover:bg-white/5 hover:text-zinc-200 transition-colors group"
        >
            <span className="text-[10px] font-mono text-zinc-700 group-hover:text-zinc-500 transition-colors">{number}</span>
            <span className="truncate">{title}</span>
        </Link>
    )
}
