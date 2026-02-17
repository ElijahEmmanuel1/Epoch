import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { ArrowLeft, Search } from "lucide-react";

export default function NotFound() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-background px-6">
            <div className="text-center max-w-md mx-auto space-y-6">
                <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-muted border border-border mx-auto">
                    <Search className="h-7 w-7 text-muted-foreground" />
                </div>

                <div className="space-y-2">
                    <h1 className="text-5xl font-bold tracking-tight text-foreground">404</h1>
                    <p className="text-lg text-muted-foreground font-serif">
                        This page doesn&apos;t exist. It might have been moved or deleted.
                    </p>
                </div>

                <div className="flex items-center justify-center gap-3">
                    <Link href="/">
                        <Button variant="primary" size="md">
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            Back Home
                        </Button>
                    </Link>
                    <Link href="/dashboard">
                        <Button variant="secondary" size="md">
                            Dashboard
                        </Button>
                    </Link>
                </div>
            </div>
        </div>
    );
}
