"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, Home, Layers, Zap, GraduationCap, Brain, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/Progress";
import { Avatar } from "@/components/ui/Avatar";
import { ThemeToggle } from "@/components/layout/ThemeToggle";

const navigation = [
    { name: "Overview", href: "/dashboard", icon: Home },
    { name: "The Fundamentals", href: "/chapter/1", icon: Layers },
    { name: "Training & Optimization", href: "/chapter/5", icon: Zap },
    { name: "Architectures", href: "/chapter/10", icon: Brain },
];

const syllabus = [
    { href: "/chapter/1", number: "01", title: "Introduction" },
    { href: "/chapter/2", number: "02", title: "Linear Regression" },
    { href: "/chapter/3", number: "03", title: "Shallow Networks" },
    { href: "/chapter/4", number: "04", title: "Deep Networks" },
    { href: "/chapter/5", number: "05", title: "Loss Functions" },
    { href: "/chapter/6", number: "06", title: "Gradient Descent" },
    { href: "/chapter/7", number: "07", title: "Backpropagation" },
    { href: "/chapter/8", number: "08", title: "Regularization" },
    { href: "/chapter/9", number: "09", title: "Convolutional Nets" },
    { href: "/chapter/10", number: "10", title: "Transformers" },
];

export function SidebarContent() {
    const pathname = usePathname();

    return (
        <div className="flex flex-col h-full">
            {/* Logo */}
            <div className="p-6 border-b border-border">
                <Link href="/" className="flex items-center gap-2.5 group">
                    <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-foreground to-foreground/70 flex items-center justify-center shadow-sm">
                        <span className="text-background font-bold text-xs">E</span>
                    </div>
                    <span className="font-bold text-sm tracking-tight text-foreground">Epoch</span>
                </Link>
            </div>

            {/* Navigation */}
            <div className="flex-1 py-6 px-3 space-y-8 overflow-y-auto scrollbar-thin">
                {/* Main Navigation */}
                <div className="space-y-1">
                    {navigation.map((item) => {
                        const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-xl transition-all group",
                                    isActive
                                        ? "bg-foreground/10 text-foreground shadow-sm"
                                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                                )}
                                aria-current={isActive ? "page" : undefined}
                            >
                                <item.icon className={cn("h-4 w-4 shrink-0 transition-colors", isActive ? "text-foreground" : "text-muted-foreground group-hover:text-foreground")} />
                                <span className="truncate">{item.name}</span>
                                {isActive && (
                                    <ChevronRight className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
                                )}
                            </Link>
                        );
                    })}
                </div>

                {/* Syllabus */}
                <div>
                    <h3 className="px-3 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                        <GraduationCap className="h-3.5 w-3.5" />
                        Syllabus
                    </h3>
                    <div className="space-y-0.5">
                        {syllabus.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    className={cn(
                                        "flex items-center gap-3 px-3 py-2 text-sm rounded-xl transition-all group",
                                        isActive
                                            ? "bg-foreground/10 text-foreground"
                                            : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                                    )}
                                    aria-current={isActive ? "page" : undefined}
                                >
                                    <span className={cn(
                                        "text-[10px] font-mono tabular-nums transition-colors",
                                        isActive ? "text-foreground" : "text-muted-foreground/50 group-hover:text-muted-foreground"
                                    )}>
                                        {item.number}
                                    </span>
                                    <span className="truncate">{item.title}</span>
                                </Link>
                            );
                        })}
                    </div>
                </div>

                {/* Progress summary */}
                <div className="px-3">
                    <div className="rounded-xl border border-border bg-muted/30 p-4 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-foreground">Your Progress</span>
                            <span className="text-[10px] text-muted-foreground">2/10</span>
                        </div>
                        <Progress value={20} variant="gradient" size="sm" />
                        <p className="text-[11px] text-muted-foreground">
                            Continue with Chapter 3 to stay on track.
                        </p>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-border space-y-2">
                <div className="flex items-center justify-between px-1">
                    <ThemeToggle />
                </div>
                <div className="flex items-center gap-3 p-2.5 rounded-xl bg-muted/30 border border-border">
                    <Avatar fallback="US" size="sm" status="online" />
                    <div className="flex-1 overflow-hidden">
                        <p className="text-xs font-medium text-foreground truncate">User Account</p>
                        <p className="text-[10px] text-muted-foreground truncate">Pro Plan</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
