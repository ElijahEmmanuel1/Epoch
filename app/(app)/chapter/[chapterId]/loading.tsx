import { Skeleton } from "@/components/ui/Skeleton";

export default function ChapterLoading() {
    return (
        <div className="max-w-3xl mx-auto space-y-8 animate-fade-in">
            {/* Breadcrumb skeleton */}
            <Skeleton className="h-4 w-48" />

            {/* Header skeleton */}
            <div className="space-y-4">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-10 w-72" />
                <Skeleton className="h-5 w-full max-w-lg" />
            </div>

            {/* Content skeleton */}
            <div className="space-y-6 pt-4">
                <div className="space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                </div>

                {/* Math block skeleton */}
                <Skeleton variant="rectangular" className="h-32 w-full" />

                <div className="space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-2/3" />
                </div>
            </div>
        </div>
    );
}
