import { Sidebar } from "@/components/layout/Sidebar";

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="flex min-h-screen flex-col md:flex-row">
            <Sidebar className="hidden md:block" />
            <main className="flex-1 p-8">
                {children}
            </main>
        </div>
    );
}
