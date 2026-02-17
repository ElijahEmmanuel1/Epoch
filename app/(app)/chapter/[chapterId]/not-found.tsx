import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { ArrowLeft, BookOpen } from "lucide-react";

export default function ChapterNotFound() {
    return (
        <div className="max-w-md mx-auto text-center py-24 space-y-6">
            <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-muted border border-border mx-auto">
                <BookOpen className="h-7 w-7 text-muted-foreground" />
            </div>

            <div className="space-y-2">
                <h1 className="text-2xl font-bold tracking-tight text-foreground">Chapter Not Found</h1>
                <p className="text-muted-foreground font-serif">
                    This chapter doesn&apos;t exist or hasn&apos;t been published yet.
                </p>
            </div>

            <Link href="/dashboard">
                <Button variant="primary" size="md">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to Dashboard
                </Button>
            </Link>
        </div>
    );
}
