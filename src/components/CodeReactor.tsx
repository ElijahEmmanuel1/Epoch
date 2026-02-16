import { useState, useRef, useCallback } from 'react'
import Editor, { type OnMount } from '@monaco-editor/react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play, RotateCcw, Eye, EyeOff, ChevronUp, ChevronDown,
  Check, Copy, Sparkles
} from 'lucide-react'
import type { CourseNode, Exercise } from '../data/courses'
import TensorMonitor from './TensorMonitor'
import styles from './CodeReactor.module.css'

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
  course,
  activeTab,
  currentExercise,
  onHoverVar,
  updateCourse,
  activeExercise,
  setActiveExercise,
}: Props) {
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

  // Update code when tab/exercise changes
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

    // Hover-to-Connect: detect variable hover
    editor.onDidChangeCursorPosition((e) => {
      const model = editor.getModel()
      if (!model) return
      const word = model.getWordAtPosition(e.position)
      if (word) {
        onHoverVar(word.word)
      }
    })
  }

  const runCode = useCallback(() => {
    setIsRunning(true)
    setConsoleOutput([])
    setConsoleCollapsed(false)

    // Simulate code execution with parsed output
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
    if (!showSolution) {
      setCode(currentExercise.solution)
    } else {
      setCode(currentExercise.starterCode)
    }
    setShowSolution(!showSolution)
  }

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const markComplete = () => {
    if (currentExercise) {
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
  }

  return (
    <div className={styles.reactor}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <span className={styles.fileLabel}>
            <span className={styles.fileDot} />
            {activeTab === 'exercise' ? `exercise_${activeExercise + 1}.py` : `${course.id}.py`}
          </span>
        </div>
        <div className={styles.toolbarRight}>
          {activeTab === 'exercise' && currentExercise && (
            <>
              <button
                className={styles.toolBtn}
                onClick={toggleSolution}
                title={showSolution ? 'Masquer la solution' : 'Voir la solution'}
              >
                {showSolution ? <EyeOff size={14} /> : <Eye size={14} />}
                <span>{showSolution ? 'Masquer' : 'Solution'}</span>
              </button>
              <button className={styles.toolBtn} onClick={markComplete}>
                <Check size={14} />
                <span>Valider</span>
              </button>
            </>
          )}
          <button className={styles.toolBtn} onClick={copyCode}>
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <button className={styles.toolBtn} onClick={resetCode}>
            <RotateCcw size={14} />
          </button>
          <motion.button
            className={styles.runBtn}
            onClick={runCode}
            disabled={isRunning}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isRunning ? (
              <Sparkles size={16} className={styles.spinning} />
            ) : (
              <Play size={16} />
            )}
            <span>RUN EPOCH</span>
          </motion.button>
        </div>
      </div>

      {/* Editor */}
      <div className={styles.editorWrapper}>
        <Editor
          height="100%"
          language="python"
          theme="epoch-dark"
          value={code}
          onChange={(value) => setCode(value || '')}
          onMount={handleEditorMount}
          beforeMount={(monaco) => {
            monaco.editor.defineTheme('epoch-dark', {
              base: 'vs-dark',
              inherit: true,
              rules: [
                { token: 'comment', foreground: '5c6585', fontStyle: 'italic' },
                { token: 'keyword', foreground: 'd946ef' },
                { token: 'string', foreground: '39ffb0' },
                { token: 'number', foreground: 'ff9100' },
                { token: 'type', foreground: '00e5ff' },
                { token: 'function', foreground: '00e5ff' },
                { token: 'variable', foreground: 'e8eaf6' },
                { token: 'operator', foreground: 'd946ef' },
                { token: 'delimiter', foreground: '9fa8c7' },
              ],
              colors: {
                'editor.background': '#0d1120',
                'editor.foreground': '#e8eaf6',
                'editor.lineHighlightBackground': '#161b3366',
                'editor.selectionBackground': '#00e5ff33',
                'editorLineNumber.foreground': '#2a3366',
                'editorLineNumber.activeForeground': '#5c6585',
                'editor.inactiveSelectionBackground': '#161b3344',
                'editorIndentGuide.background': '#1e254522',
                'editorCursor.foreground': '#00e5ff',
                'editorWhitespace.foreground': '#1e254544',
              },
            })
          }}
          options={{
            fontSize: 14,
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

      {/* Console / Tensor Monitor */}
      <div className={`${styles.console} ${consoleCollapsed ? styles.consoleCollapsed : ''}`}>
        <button
          className={styles.consoleHeader}
          onClick={() => setConsoleCollapsed(!consoleCollapsed)}
        >
          <span className={styles.consoleTitle}>
            <span className={styles.consoleDot} data-state={consoleOutput.length > 0 ? 'active' : 'idle'} />
            TENSOR MONITOR
          </span>
          {consoleCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        <AnimatePresence>
          {!consoleCollapsed && (
            <motion.div
              className={styles.consoleBody}
              initial={{ height: 0 }}
              animate={{ height: 200 }}
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

// ── Console Entry types ──
export interface ConsoleEntry {
  type: 'output' | 'tensor' | 'error' | 'info'
  text: string
  shape?: string
  values?: number[][]
}

// ── Simulated Python execution ──
function simulateExecution(code: string): ConsoleEntry[] {
  const entries: ConsoleEntry[] = []

  // Parse print statements from code
  const lines = code.split('\n')

  for (const line of lines) {
    const trimmed = line.trim()

    // Skip comments and empty lines
    if (trimmed.startsWith('#') || trimmed === '') continue

    // Detect print statements
    if (trimmed.startsWith('print(')) {
      const content = trimmed.match(/print\((.*)\)/)
      if (content) {
        let text = content[1]
        // Clean up f-string formatting
        text = text.replace(/^f"/, '').replace(/"$/, '')
        text = text.replace(/^f'/, '').replace(/'$/, '')
        text = text.replace(/^"/, '').replace(/"$/, '')
        text = text.replace(/^'/, '').replace(/'$/, '')

        // Detect tensor shape patterns
        if (text.includes('.shape') || text.toLowerCase().includes('shape')) {
          const shapes = ['[3, 4]', '[4, 2]', '[3, 2]', '[32, 128]', '[1, 10]', '[3, 224, 224]']
          const shape = shapes[Math.floor(Math.random() * shapes.length)]
          entries.push({
            type: 'tensor',
            text: text.replace(/\{.*?\}/g, shape),
            shape,
            values: generateHeatmapData(3, 4),
          })
        } else if (text.includes('loss') || text.includes('Loss')) {
          const lossVal = (Math.random() * 2 + 0.1).toFixed(4)
          entries.push({
            type: 'output',
            text: text.replace(/\{.*?\}/g, lossVal),
          })
        } else {
          // General output with simulated values
          let output = text.replace(/\{.*?\.item\(\).*?\}/g, () => (Math.random() * 10 - 5).toFixed(4))
          output = output.replace(/\{.*?\}/g, () => `[${(Math.random()*3).toFixed(2)}, ${(Math.random()*3).toFixed(2)}]`)
          entries.push({ type: 'output', text: output })
        }
      }
    }
  }

  if (entries.length === 0) {
    entries.push({ type: 'info', text: '✓ Code exécuté avec succès (aucune sortie)' })
  }

  // Check for common errors
  if (code.includes('___')) {
    return [{ type: 'error', text: 'SyntaxError: expression attendue — remplacez les ___ par votre code' }]
  }

  return entries
}

function generateHeatmapData(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.random() * 2 - 1)
  )
}
