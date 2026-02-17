export default function DashboardPage() {
    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold tracking-tight mb-4">Syllabus</h1>
            <p className="text-muted-foreground mb-8">
                Pick up where you left off or start a new module.
            </p>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {/* Placeholder cards */}
                {[1, 2, 3, 4].map((chapter) => (
                    <div key={chapter} className="rounded-xl border bg-card text-card-foreground shadow-sm p-6 hover:shadow-md transition-shadow cursor-pointer">
                        <div className="flex flex-col gap-1">
                            <span className="text-xs font-medium text-muted-foreground">Chapter {chapter}</span>
                            <h3 className="font-semibold leading-none tracking-tight">
                                {chapter === 1 ? "Introduction" : chapter === 2 ? "Linear Regression" : chapter === 3 ? "Shallow Networks" : "Deep Networks"}
                            </h3>
                            <p className="text-sm text-muted-foreground mt-2">
                                {chapter === 1 ? "Basics of machine learning." : "Fitting lines to data."}
                            </p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
