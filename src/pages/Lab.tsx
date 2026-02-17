import { useParams, useNavigate } from 'react-router-dom'
import { useState, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowLeft, BookOpen, FlaskConical, ChevronRight, ChevronLeft, ChevronDown, ChevronUp } from 'lucide-react'
import type { CourseNode } from '../data/courses'
import NeuralCanvas from '../components/NeuralCanvas'
import CodeReactor from '../components/CodeReactor'
import styles from './Lab.module.css'

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
    const handler = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [])

  const course = useMemo(
    () => courses.find(c => c.id === courseId),
    [courses, courseId]
  )

  if (!course) {
    return (
      <div className={styles.notFound}>
        <h2>Module non trouvé</h2>
        <button onClick={() => navigate('/')}>Retour à la Roadmap</button>
      </div>
    )
  }

  // Mark as in-progress on first visit
  if (course.status === 'available') {
    updateCourse(course.id, { status: 'in-progress' })
  }

  const currentExercise = course.exercises[activeExercise]

  return (
    <div className={styles.container}>
      {/* Top Bar */}
      <header className={styles.topbar}>
        <button className={styles.backBtn} onClick={() => navigate('/')}>
          <ArrowLeft size={16} />
          <span>Roadmap</span>
        </button>
        <div className={styles.topbarCenter}>
          <h2 className={styles.courseTitle}>{course.title}</h2>
          <span className={styles.courseBadge}>{course.category}</span>
        </div>
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${activeTab === 'theory' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('theory')}
          >
            <BookOpen size={14} />
            Théorie
          </button>
          {course.exercises.length > 0 && (
            <button
              className={`${styles.tab} ${activeTab === 'exercise' ? styles.activeTab : ''}`}
              onClick={() => setActiveTab('exercise')}
            >
              <FlaskConical size={14} />
              Exercice {activeExercise + 1}/{course.exercises.length}
            </button>
          )}
        </div>
      </header>

      {/* Split Screen */}
      <div className={styles.splitScreen}>
        {/* Left Panel - Neural Canvas */}
        <AnimatePresence mode="wait">
          {!leftCollapsed && (
            <motion.div
              className={styles.leftPanel}
              initial={{ [isMobile ? 'height' : 'width']: 0, opacity: 0 }}
              animate={{
                [isMobile ? 'height' : 'width']: isMobile ? '45vh' : '45%',
                opacity: 1,
              }}
              exit={{ [isMobile ? 'height' : 'width']: 0, opacity: 0 }}
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

        {/* Divider */}
        <button
          className={styles.divider}
          onClick={() => setLeftCollapsed(!leftCollapsed)}
          title={leftCollapsed ? 'Afficher la théorie' : 'Masquer la théorie'}
        >
          <div className={styles.dividerHandle}>
            {isMobile ? (
              leftCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />
            ) : (
              leftCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />
            )}
          </div>
        </button>

        {/* Right Panel - Code Reactor */}
        <div className={styles.rightPanel}>
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
