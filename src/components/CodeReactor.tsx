import { useState, useRef, useCallback } from 'react'
import Editor, { type OnMount } from '@monaco-editor/react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play, RotateCcw, Eye, EyeOff, ChevronUp, ChevronDown,
  Check, Copy, Sparkles
} from 'lucide-react'
import { clsx } from 'clsx'
import type { CourseNode, Exercise } from '../data/courses'
import TensorMonitor from './TensorMonitor'
import { useTheme } from '../contexts/ThemeContext'

interface Props {
  course: CourseNode
  activeTab: 'theory' | 'exercise'
  currentExercise?: Exercise
  onHoverVar: (varName: string | null) => void
  updateCourse: (id: string, partial: Partial<CourseNode>) => void
  activeExercise: number
  setActiveExercise: (i: number) => void
}

export default function CodeReactor({
  course, activeTab, currentExercise, onHoverVar,
  updateCourse, activeExercise, setActiveExercise,
}: Props) {
  const { theme } = useTheme()

  const initialCode = activeTab === 'exercise' && currentExercise
    ? currentExercise.starterCode
    : course.codeTemplate

  const [code, setCode] = useState(initialCode)
  const [consoleOutput, setConsoleOutput] = useState<ConsoleEntry[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [showSolution, setShowSolution] = useState(false)
  const [consoleCollapsed, setConsoleCollapsed] = useState(false)
  const [copied, setCopied] = useState(false)
  const editorRef = useRef<any>(null)

  const prevTabRef = useRef(activeTab)
  const prevExRef = useRef(activeExercise)
  if (prevTabRef.current !== activeTab || prevExRef.current !== activeExercise) {
    prevTabRef.current = activeTab
    prevExRef.current = activeExercise
    const newCode = activeTab === 'exercise' && currentExercise
      ? currentExercise.starterCode
      : course.codeTemplate
    setCode(newCode)
    setShowSolution(false)
    setConsoleOutput([])
  }

  const handleEditorMount: OnMount = (editor) => {
    editorRef.current = editor
    editor.onDidChangeCursorPosition((e) => {
      const model = editor.getModel()
      if (!model) return
      const word = model.getWordAtPosition(e.position)
      if (word) onHoverVar(word.word)
    })
  }

  const runCode = useCallback(() => {
    setIsRunning(true)
    setConsoleOutput([])
    setConsoleCollapsed(false)
    setTimeout(() => {
      const output = simulateExecution(code)
      setConsoleOutput(output)
      setIsRunning(false)
    }, 800)
  }, [code])

  const resetCode = () => {
    const newCode = activeTab === 'exercise' && currentExercise
      ? currentExercise.starterCode
      : course.codeTemplate
    setCode(newCode)
    setConsoleOutput([])
    setShowSolution(false)
  }

  const toggleSolution = () => {
    if (!currentExercise) return
    setCode(showSolution ? currentExercise.starterCode : currentExercise.solution)
    setShowSolution(!showSolution)
  }

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const markComplete = () => {
    if (!currentExercise) return
    const updatedExercises = course.exercises.map((ex, i) =>
      i === activeExercise ? { ...ex, completed: true } : ex
    )
    const allComplete = updatedExercises.every(ex => ex.completed)
    updateCourse(course.id, {
      exercises: updatedExercises,
      status: allComplete ? 'completed' : 'in-progress',
      progress: Math.round((updatedExercises.filter(e => e.completed).length / updatedExercises.length) * 100),
    })
    if (activeExercise < course.exercises.length - 1) {
      setActiveExercise(activeExercise + 1)
    }
  }

  const editorTheme = theme === 'dark' ? 'epoch-dark' : 'epoch-light'

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 md:px-4 py-2 bg-[var(--bg-surface)] border-b border-[var(--border-default)] shrink-0 transition-colors">
        <div className="flex items-center">
          <span className="flex items-center gap-2 font-mono text-xs text-[var(--text-secondary)]">
            <span className="w-2 h-2 rounded-full bg-[var(--color-success)]" />
            {activeTab === 'exercise' ? `exercise_${activeExercise + 1}.py` : `${course.id}.py`}
          </span>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {activeTab === 'exercise' && currentExercise && (
            <>
              <button
                className="flex items-center gap-1.5 px-2.5 py-1.5 border border-[var(--border-default)] rounded-md text-[var(--text-muted)] text-[11px] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)] transition-all"
                onClick={toggleSolution}
              >
                {showSolution ? <EyeOff size={13} /> : <Eye size={13} />}
                <span className="hidden sm:inline">{showSolution ? 'Masquer' : 'Solution'}</span>
              </button>
              <button
                className="flex items-center gap-1.5 px-2.5 py-1.5 border border-[var(--border-default)] rounded-md text-[var(--text-muted)] text-[11px] hover:bg-[var(--bg-hover)] hover:text-[var(--color-success)] transition-all"
                onClick={markComplete}
              >
                <Check size={13} />
                <span className="hidden sm:inline">Valider</span>
              </button>
            </>
          )}
          <button
            className="flex items-center justify-center w-8 h-8 border border-[var(--border-default)] rounded-md text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)] transition-all"
            onClick={copyCode}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
          <button
            className="flex items-center justify-center w-8 h-8 border border-[var(--border-default)] rounded-md text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)] transition-all"
            onClick={resetCode}
          >
            <RotateCcw size={13} />
          </button>
          <motion.button
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-gradient-to-r from-primary to-accent text-white font-mono text-xs font-bold tracking-wide disabled:opacity-60 disabled:cursor-wait"
            onClick={runCode}
            disabled={isRunning}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            {isRunning ? <Sparkles size={14} className="animate-spin" /> : <Play size={14} />}
            <span>RUN</span>
          </motion.button>
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <Editor
          height="100%"
          language="python"
          theme={editorTheme}
          value={code}
          onChange={(value) => setCode(value || '')}
          onMount={handleEditorMount}
          beforeMount={(monaco) => {
            monaco.editor.defineTheme('epoch-dark', {
              base: 'vs-dark',
              inherit: true,
              rules: [
                { token: 'comment', foreground: '6b6f85', fontStyle: 'italic' },
                { token: 'keyword', foreground: '818cf8' },
                { token: 'string', foreground: '34d399' },
                { token: 'number', foreground: 'f59e0b' },
                { token: 'type', foreground: '22d3ee' },
                { token: 'function', foreground: '6366f1' },
                { token: 'variable', foreground: 'e4e6f0' },
                { token: 'operator', foreground: '818cf8' },
                { token: 'delimiter', foreground: 'a0a4b8' },
              ],
              colors: {
                'editor.background': '#0f1117',
                'editor.foreground': '#e4e6f0',
                'editor.lineHighlightBackground': '#1e203022',
                'editor.selectionBackground': '#6366f133',
                'editorLineNumber.foreground': '#2a2d42',
                'editorLineNumber.activeForeground': '#6b6f85',
                'editor.inactiveSelectionBackground': '#1e203022',
                'editorIndentGuide.background': '#1e203020',
                'editorCursor.foreground': '#6366f1',
                'editorWhitespace.foreground': '#1e203044',
              },
            })
            monaco.editor.defineTheme('epoch-light', {
              base: 'vs',
              inherit: true,
              rules: [
                { token: 'comment', foreground: '8b8fa5', fontStyle: 'italic' },
                { token: 'keyword', foreground: '6366f1' },
                { token: 'string', foreground: '059669' },
                { token: 'number', foreground: 'd97706' },
                { token: 'type', foreground: '0891b2' },
                { token: 'function', foreground: '4f46e5' },
                { token: 'variable', foreground: '1a1c2e' },
                { token: 'operator', foreground: '6366f1' },
                { token: 'delimiter', foreground: '4a4d65' },
              ],
              colors: {
                'editor.background': '#f8f9fc',
                'editor.foreground': '#1a1c2e',
                'editor.lineHighlightBackground': '#f1f3f910',
                'editor.selectionBackground': '#6366f122',
                'editorLineNumber.foreground': '#c4c6d4',
                'editorLineNumber.activeForeground': '#8b8fa5',
                'editor.inactiveSelectionBackground': '#6366f111',
                'editorIndentGuide.background': '#e0e2ec55',
                'editorCursor.foreground': '#6366f1',
                'editorWhitespace.foreground': '#e0e2ec',
              },
            })
          }}
          options={{
            fontSize: 13,
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            fontLigatures: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            lineNumbers: 'on',
            glyphMargin: false,
            folding: true,
            lineDecorationsWidth: 12,
            lineNumbersMinChars: 3,
            renderLineHighlight: 'line',
            renderWhitespace: 'none',
            smoothScrolling: true,
            cursorBlinking: 'smooth',
            cursorSmoothCaretAnimation: 'on',
            padding: { top: 16, bottom: 16 },
            suggest: { showWords: false },
            tabSize: 4,
          }}
        />
      </div>

      {/* Console — Tensor Monitor */}
      <div className={clsx(
        'shrink-0 border-t border-[var(--border-default)] bg-[var(--bg-surface)] transition-colors',
        consoleCollapsed && 'h-auto'
      )}>
        <button
          className="flex items-center justify-between w-full px-4 py-2.5 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          onClick={() => setConsoleCollapsed(!consoleCollapsed)}
        >
          <span className="flex items-center gap-2 font-mono text-[10px] font-bold tracking-widest uppercase">
            <span className={clsx(
              'w-1.5 h-1.5 rounded-full',
              consoleOutput.length > 0 ? 'bg-[var(--color-success)] shadow-[0_0_6px_var(--color-success)]' : 'bg-[var(--text-muted)]'
            )} />
            Tensor Monitor
          </span>
          {consoleCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        <AnimatePresence>
          {!consoleCollapsed && (
            <motion.div
              className="overflow-y-auto overflow-x-hidden"
              initial={{ height: 0 }}
              animate={{ height: 180 }}
              exit={{ height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <TensorMonitor entries={consoleOutput} isRunning={isRunning} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

/* ── Types ── */
export interface ConsoleEntry {
  type: 'output' | 'tensor' | 'error' | 'info'
  text: string
  shape?: string
  values?: number[][]
}

/* ── Simulation ── */
function simulateExecution(code: string): ConsoleEntry[] {
  const entries: ConsoleEntry[] = []
  const lines = code.split('\n')

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('#') || trimmed === '') continue
    if (trimmed.startsWith('print(')) {
      const content = trimmed.match(/print\((.*)\)/)
      if (content) {
        let text = content[1]
          .replace(/^f?["']/, '').replace(/["']$/, '')
        if (text.includes('.shape') || text.toLowerCase().includes('shape')) {
          const shapes = ['[3, 4]', '[4, 2]', '[3, 2]', '[32, 128]', '[1, 10]']
          const shape = shapes[Math.floor(Math.random() * shapes.length)]
          entries.push({
            type: 'tensor', text: text.replace(/\{.*?\}/g, shape), shape,
            values: Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => Math.random() * 2 - 1)),
          })
        } else if (text.includes('loss') || text.includes('Loss')) {
          entries.push({ type: 'output', text: text.replace(/\{.*?\}/g, (Math.random() * 2 + 0.1).toFixed(4)) })
        } else {
          let output = text.replace(/\{.*?\.item\(\).*?\}/g, () => (Math.random() * 10 - 5).toFixed(4))
          output = output.replace(/\{.*?\}/g, () => `[${(Math.random()*3).toFixed(2)}, ${(Math.random()*3).toFixed(2)}]`)
          entries.push({ type: 'output', text: output })
        }
      }
    }
  }

  if (entries.length === 0) entries.push({ type: 'info', text: '✓ Code exécuté avec succès (aucune sortie)' })
  if (code.includes('___')) return [{ type: 'error', text: 'SyntaxError: expression attendue — remplacez les ___ par votre code' }]
  return entries
}
