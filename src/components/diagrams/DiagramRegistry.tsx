/**
 * DiagramRegistry — Maps diagramId to React SVG components
 * When a TheoryBlock has type 'diagram' + diagramId, we render the
 * component instead of the ASCII <pre> fallback.
 */
import { type ComponentType, lazy, Suspense } from 'react'

const ReluGraph = lazy(() => import('./ReluGraph'))
const SlopeConstructionGraph = lazy(() => import('./SlopeConstructionGraph'))
const BumpConstructionGraph = lazy(() => import('./BumpConstructionGraph'))
const PiecewiseLinearGraph = lazy(() => import('./PiecewiseLinearGraph'))
const ShallowNetPipeline = lazy(() => import('./ShallowNetPipeline'))
const ApproximationComparison = lazy(() => import('./ApproximationComparison'))

const registry: Record<string, ComponentType> = {
  'relu-graph': ReluGraph,
  'slope-construction': SlopeConstructionGraph,
  'bump-construction': BumpConstructionGraph,
  'piecewise-linear': PiecewiseLinearGraph,
  'shallow-net-pipeline': ShallowNetPipeline,
  'universal-approximation': ApproximationComparison,
}

interface Props {
  diagramId: string
}

export default function DiagramRenderer({ diagramId }: Props) {
  const Component = registry[diagramId]
  if (!Component) return null

  return (
    <Suspense fallback={
      <div className="p-5 text-[var(--text-muted)] font-mono text-xs">
        Chargement du diagramme…
      </div>
    }>
      <Component />
    </Suspense>
  )
}

export function hasDiagram(id: string): boolean {
  return id in registry
}
