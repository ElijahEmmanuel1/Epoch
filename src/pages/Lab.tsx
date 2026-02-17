import { useParams, useNavigate } from 'react-router-dom'
import { useState, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowLeft, BookOpen, FlaskConical, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { clsx } from 'clsx'
import type { CourseNode } from '../data/courses'
import NeuralCanvas from '../components/NeuralCanvas'
import CodeReactor from '../components/CodeReactor'

interface Props {
  courses: CourseNode[]
  updateCourse: (id: string, partial: Partial<CourseNode>) => void
}

export default function Lab({ courses, updateCourse }: Props) {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'theory' | 'exercise'>('theory')
  const [activeExercise, setActiveExercise] = useState(0)
  const [highlightedVar, setHighlightedVar] = useState<string | null>(null)
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const h = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', h)
    return () => window.removeEventListener('resize', h)
  }, [])

  const course = useMemo(
    () => courses.find(c => c.id === courseId),
    [courses, courseId]
  )

  if (!course) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-[var(--text-secondary)]">
        <h2 className="text-lg font-semibold">Module non trouvé</h2>
        <button
          onClick={() => navigate('/')}
          className="px-5 py-2.5 bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-lg text-primary hover:border-[var(--border-active)] transition-all"
        >
          Retour à la Roadmap
        </button>
      </div>
    )
  }

  if (course.status === 'available') {
    updateCourse(course.id, { status: 'in-progress' })
  }

  const currentExercise = course.exercises[activeExercise]

  return (
    <div className="flex flex-col h-screen bg-[var(--bg-base)] overflow-hidden transition-colors duration-300">
      {/* Top Bar */}
      <header className="flex flex-wrap items-center gap-2 md:gap-4 px-3 md:px-5 py-2.5 bg-[var(--bg-surface)] border-b border-[var(--border-default)] shrink-0 transition-colors">
        <div className="pl-10 lg:pl-0">
          <button
            className="flex items-center gap-1.5 px-3 py-1.5 border border-[var(--border-default)] rounded-lg text-[var(--text-secondary)] text-xs hover:border-[var(--border-active)] hover:text-primary transition-all"
            onClick={() => navigate('/')}
          >
            <ArrowLeft size={14} />
            <span className="hidden sm:inline">Roadmap</span>
          </button>
        </div>

        <div className="flex-1 min-w-0 flex items-center gap-2.5">
          <h2 className="text-sm md:text-[15px] font-bold text-[var(--text-primary)] truncate">
            {course.title}
          </h2>
          <span className="hidden md:inline text-[9px] font-bold tracking-widest uppercase text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20">
            {course.category}
          </span>
        </div>

        <div className="flex gap-1 w-full sm:w-auto order-3 sm:order-none">
          <button
            className={clsx(
              'flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3 py-2 border rounded-lg text-xs font-medium transition-all',
              activeTab === 'theory'
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'border-[var(--border-default)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--border-default)]'
            )}
            onClick={() => setActiveTab('theory')}
          >
            <BookOpen size={14} />
            Théorie
          </button>
          {course.exercises.length > 0 && (
            <button
              className={clsx(
                'flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3 py-2 border rounded-lg text-xs font-medium transition-all',
                activeTab === 'exercise'
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'border-[var(--border-default)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--border-default)]'
              )}
              onClick={() => setActiveTab('exercise')}
            >
              <FlaskConical size={14} />
              Exercice {activeExercise + 1}/{course.exercises.length}
            </button>
          )}
        </div>
      </header>

      {/* Split Screen */}
      <div className={clsx('flex-1 flex overflow-hidden', isMobile && 'flex-col')}>
        {/* Left Panel — Theory/Exercise */}
        <AnimatePresence mode="wait">
          {!leftCollapsed && (
            <motion.div
              className={clsx(
                'overflow-y-auto bg-[var(--bg-base)] shrink-0 transition-colors',
                isMobile ? 'border-b border-[var(--border-default)]' : 'border-r border-[var(--border-default)]'
              )}
              initial={isMobile ? { height: 0, opacity: 0 } : { width: 0, opacity: 0 }}
              animate={isMobile ? { height: '45vh', opacity: 1 } : { width: '46%', opacity: 1 }}
              exit={isMobile ? { height: 0, opacity: 0 } : { width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            >
              <NeuralCanvas
                course={course}
                activeTab={activeTab}
                currentExercise={currentExercise}
                highlightedVar={highlightedVar}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Divider Toggle */}
        <button
          className={clsx(
            'flex items-center justify-center bg-[var(--bg-surface)] border-[var(--border-default)] hover:bg-[var(--bg-hover)] transition-all shrink-0',
            isMobile
              ? 'w-full h-8 border-b'
              : 'w-3 border-r cursor-col-resize'
          )}
          onClick={() => setLeftCollapsed(!leftCollapsed)}
          title={leftCollapsed ? 'Afficher la théorie' : 'Masquer la théorie'}
        >
          {leftCollapsed ? (
            <PanelLeftOpen size={14} className="text-[var(--text-muted)]" />
          ) : (
            <PanelLeftClose size={14} className="text-[var(--text-muted)]" />
          )}
        </button>

        {/* Right Panel — Code Editor */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0 min-h-0">
          <CodeReactor
            course={course}
            activeTab={activeTab}
            currentExercise={currentExercise}
            onHoverVar={setHighlightedVar}
            updateCourse={updateCourse}
            activeExercise={activeExercise}
            setActiveExercise={setActiveExercise}
          />
        </div>
      </div>
    </div>
  )
}
