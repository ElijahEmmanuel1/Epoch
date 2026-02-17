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
        <div className={cn("pb-12 min-h-screen w-64 border-r bg-background", className)}>
            <div className="space-y-4 py-4">
                <div className="px-3 py-2">
                    <Link href="/" className="mb-2 px-4 text-xl font-bold tracking-tight flex items-center">
                        Epoch
                    </Link>
                    <div className="space-y-1 mt-4">
                        {navigation.map((item) => (
                            <Link
                                key={item.name}
                                href={item.href}
                                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted/50 transition-colors"
                            >
                                <item.icon className="h-4 w-4" />
                                {item.name}
                            </Link>
                        ))}
                    </div>
                </div>
                <div className="px-3 py-2">
                    <h2 className="mb-2 px-4 text-xs font-semibold tracking-tight text-muted-foreground uppercase">
                        Modules
                    </h2>
                    <div className="space-y-1">
                        <Link href="/chapter/1" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted/50 transition-colors">
                            1. Introduction
                        </Link>
                        <Link href="/chapter/2" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted/50 transition-colors">
                            2. Linear Regression
                        </Link>
                        <Link href="/chapter/3" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted/50 transition-colors">
                            3. Shallow Nets
                        </Link>
                        <Link href="/chapter/4" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted/50 transition-colors">
                            4. Deep Nets
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}
