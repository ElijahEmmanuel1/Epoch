import { SidebarContent } from "@/components/layout/SidebarContent";
import { MobileNav } from "@/components/layout/MobileNav";
import type { Metadata } from "next";

export const metadata: Metadata = {
    title: {
        template: "%s | Epoch",
        default: "Dashboard | Epoch",
    },
};

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="flex min-h-screen">
            {/* Desktop Sidebar */}
            <aside className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0 border-r border-border bg-card/50 backdrop-blur-xl">
                <SidebarContent />
            </aside>

            {/* Mobile Navigation */}
            <MobileNav>
                <SidebarContent />
            </MobileNav>

            {/* Main content */}
            <main className="flex-1 md:pl-64">
                <div className="p-6 md:p-10 max-w-6xl mx-auto">
                    {children}
                </div>
            </main>
        </div>
    );
}
