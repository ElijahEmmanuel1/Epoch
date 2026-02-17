import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'
import { Lightbulb, AlertTriangle, Info, HelpCircle, Terminal, X, Play, Sparkles } from 'lucide-react'
import { clsx } from 'clsx'
import type { CourseNode, Exercise, TheoryBlock } from '../data/courses'
import NeuralDiagram from './NeuralDiagram'
import DiagramRenderer, { hasDiagram } from './diagrams/DiagramRegistry'

interface Props {
  course: CourseNode
  activeTab: 'theory' | 'exercise'
  currentExercise?: Exercise
  highlightedVar: string | null
}

export default function NeuralCanvas({ course, activeTab, currentExercise, highlightedVar }: Props) {
  return (
    <div className="h-full overflow-y-auto">
      {activeTab === 'theory' ? (
        <TheoryView theory={course.theory} highlightedVar={highlightedVar} courseId={course.id} />
      ) : currentExercise ? (
        <ExerciseView exercise={currentExercise} />
      ) : (
        <div className="flex flex-col items-center justify-center h-48 gap-3 text-[var(--text-muted)]">
          <Info size={24} />
          <p className="text-sm">Aucun exercice pour ce module</p>
        </div>
      )}
    </div>
  )
}

/* ════════════════════════════════
   Theory View — with inline console
   ════════════════════════════════ */
function TheoryView({
  theory, highlightedVar, courseId,
}: {
  theory: TheoryBlock[]
  highlightedVar: string | null
  courseId: string
}) {
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [consoleCode, setConsoleCode] = useState('')
  const [consoleOutput, setConsoleOutput] = useState<string[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const runConsole = useCallback(() => {
    setIsRunning(true)
    setTimeout(() => {
      const lines = consoleCode.split('\n')
      const output: string[] = []
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith('#')) continue
        const printMatch = trimmed.match(/print\((.*)\)/)
        if (printMatch) {
          let text = printMatch[1]
            .replace(/^f?["']/, '').replace(/["']$/, '')
            .replace(/\{.*?\}/g, () => (Math.random() * 10 - 5).toFixed(4))
          output.push(text)
        }
      }
      if (output.length === 0) output.push('✓ Code exécuté (aucune sortie)')
      setConsoleOutput(output)
      setIsRunning(false)
    }, 500)
  }, [consoleCode])

  return (
    <div className="p-5 md:p-6 space-y-5">
      {/* Architecture Diagram */}
      <div className="bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-xl p-4 transition-colors">
        <NeuralDiagram courseId={courseId} highlightedVar={highlightedVar} />
      </div>

      {/* Theory Blocks */}
      {theory.map((block, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04 }}
        >
          {block.type === 'text' && (
            <div className="text-sm leading-7 text-[var(--text-secondary)]">
              <MarkdownRenderer content={block.content} />
            </div>
          )}

          {block.type === 'equation' && (
            <div
              className={clsx(
                'bg-[var(--bg-surface)] border rounded-xl px-5 py-4 text-center transition-all overflow-x-auto',
                highlightedVar && block.highlightVar === highlightedVar
                  ? 'border-primary shadow-[var(--shadow-glow)]'
                  : 'border-[var(--border-default)]'
              )}
            >
              {block.label && (
                <span className="block text-[10px] font-bold tracking-widest uppercase text-[var(--text-muted)] mb-3">
                  {block.label}
                </span>
              )}
              <div className="text-base md:text-lg overflow-x-auto">
                <BlockMath math={block.content} />
              </div>
            </div>
          )}

          {block.type === 'diagram' && (
            <div className="bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-xl p-4 overflow-x-auto transition-colors">
              {block.label && (
                <span className="block text-[10px] font-bold tracking-widest uppercase text-[var(--text-muted)] mb-2.5">
                  {block.label}
                </span>
              )}
              {block.diagramId && hasDiagram(block.diagramId) ? (
                <DiagramRenderer diagramId={block.diagramId} />
              ) : (
                <pre className="font-mono text-[11px] leading-relaxed text-primary whitespace-pre overflow-x-auto">
                  {block.content}
                </pre>
              )}
            </div>
          )}

          {block.type === 'code' && (
            <div className="bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-xl overflow-hidden transition-colors">
              {block.title && (
                <div className="px-4 py-2 border-b border-[var(--border-default)] text-xs font-semibold text-[var(--text-muted)] bg-[var(--bg-elevated)]">
                  {block.title}
                </div>
              )}
              <pre className="p-4 overflow-x-auto">
                <code className="font-mono text-[12.5px] leading-relaxed text-[var(--text-primary)]">
                  {block.content}
                </code>
              </pre>
            </div>
          )}

          {block.type === 'callout' && (
            <div className="flex gap-3 bg-[var(--bg-surface)] border border-[var(--border-default)] border-l-[3px] border-l-primary rounded-r-xl p-4 transition-colors">
              <div className="text-primary shrink-0 mt-0.5">
                {block.content.startsWith('💡') ? <Lightbulb size={16} /> :
                 block.content.startsWith('⚠') ? <AlertTriangle size={16} /> :
                 block.content.startsWith('🧠') ? <HelpCircle size={16} /> :
                 <Info size={16} />}
              </div>
              <div className="text-[13px] leading-relaxed text-[var(--text-secondary)]">
                <MarkdownRenderer content={block.content.replace(/^[💡⚠🧠⚡]\s*/, '')} />
              </div>
            </div>
          )}
        </motion.div>
      ))}

      {/* Floating Console Toggle */}
      <button
        onClick={() => setConsoleOpen(!consoleOpen)}
        className={clsx(
          'fixed bottom-5 right-5 z-50 flex items-center gap-2 px-4 py-2.5 rounded-full shadow-lg transition-all duration-300',
          consoleOpen
            ? 'bg-[var(--color-error)] text-white'
            : 'bg-gradient-to-r from-primary to-accent text-white hover:shadow-xl'
        )}
      >
        {consoleOpen ? <X size={16} /> : <Terminal size={16} />}
        <span className="text-xs font-semibold">{consoleOpen ? 'Fermer' : 'Console'}</span>
      </button>

      {/* Embedded Console Panel */}
      <AnimatePresence>
        {consoleOpen && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            className="fixed bottom-16 right-5 z-50 w-[min(420px,calc(100vw-2.5rem))] bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-xl shadow-2xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--border-default)] bg-[var(--bg-elevated)]">
              <span className="text-xs font-bold tracking-wider uppercase text-[var(--text-muted)]">
                <Terminal size={12} className="inline mr-1.5 -mt-0.5" />
                Console Python
              </span>
              <button
                onClick={runConsole}
                disabled={isRunning || !consoleCode.trim()}
                className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-gradient-to-r from-primary to-accent text-white text-[11px] font-bold hover:opacity-90 disabled:opacity-50 transition-all"
              >
                {isRunning ? <Sparkles size={12} className="animate-spin" /> : <Play size={12} />}
                Exécuter
              </button>
            </div>
            <textarea
              value={consoleCode}
              onChange={e => setConsoleCode(e.target.value)}
              placeholder="# Tapez votre code Python ici…"
              className="w-full h-28 p-3 bg-[var(--bg-base)] text-[var(--text-primary)] font-mono text-xs leading-relaxed resize-none border-none outline-none placeholder:text-[var(--text-muted)]"
            />
            {consoleOutput.length > 0 && (
              <div className="border-t border-[var(--border-default)] p-3 bg-[var(--bg-elevated)] max-h-32 overflow-y-auto">
                {consoleOutput.map((line, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs font-mono py-0.5">
                    <span className="text-[var(--color-success)] shrink-0">{'>'}</span>
                    <span className="text-[var(--text-primary)]">{line}</span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ════════════════════════════════
   Exercise View
   ════════════════════════════════ */
function ExerciseView({ exercise }: { exercise: Exercise }) {
  const [showHints, setShowHints] = useState(false)
  return (
    <div className="p-5 md:p-6">
      <h3 className="text-lg font-bold text-[var(--text-primary)] mb-4">{exercise.title}</h3>
      <div className="text-sm leading-7 text-[var(--text-secondary)] mb-5">
        <MarkdownRenderer content={exercise.instructions} />
      </div>
      {exercise.hints.length > 0 && (
        <div className="mt-4">
          <button
            className="flex items-center gap-2 px-4 py-2.5 bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-lg text-[var(--color-warning)] text-xs font-medium hover:border-[var(--color-warning)]/30 hover:bg-[var(--color-warning)]/5 transition-all"
            onClick={() => setShowHints(!showHints)}
          >
            <Lightbulb size={14} />
            {showHints ? 'Masquer les indices' : `${exercise.hints.length} indice(s) disponible(s)`}
          </button>
          <AnimatePresence>
            {showHints && (
              <motion.ul
                className="mt-3 pl-5 space-y-2"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                {exercise.hints.map((hint, i) => (
                  <li key={i} className="text-[13px] text-[var(--text-secondary)] leading-relaxed list-disc">{hint}</li>
                ))}
              </motion.ul>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

/* ════════════════════════════════
   Markdown + LaTeX Parser
   ════════════════════════════════ */
function MarkdownRenderer({ content }: { content: string }) {
  const parts = content.split(/(\$[^$]+\$)/g)
  return (
    <span>
      {parts.map((part, index) => {
        if (part.startsWith('$') && part.endsWith('$')) {
          return <InlineMath key={index} math={part.slice(1, -1)} />
        }
        return <span key={index} dangerouslySetInnerHTML={{ __html: formatMarkdown(part) }} />
      })}
    </span>
  )
}

function formatMarkdown(text: string): string {
  let safe = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  return safe
    .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-[var(--text-primary)]">$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="font-mono text-xs bg-[var(--bg-overlay)] text-primary px-1.5 py-0.5 rounded border border-[var(--border-default)]">$1</code>')
    .replace(/\n/g, '<br/>')
}

