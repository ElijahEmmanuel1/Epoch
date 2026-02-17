import { useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Map, Zap, X, Sun, Moon } from 'lucide-react'
import { clsx } from 'clsx'
import type { CourseNode } from '../data/courses'
import { getTotalProgress } from '../data/courses'
import { useTheme } from '../contexts/ThemeContext'

interface Props {
  courses: CourseNode[]
  isMobile?: boolean
  isOpen?: boolean
  onClose?: () => void
}

export default function Sidebar({ courses, isMobile, isOpen, onClose }: Props) {
  const navigate = useNavigate()
  const location = useLocation()
  const progress = getTotalProgress(courses)
  const { theme, toggleTheme } = useTheme()

  const handleNav = (path: string) => {
    navigate(path)
    if (isMobile && onClose) onClose()
  }

  if (isMobile && !isOpen) return null

  const statusColors: Record<string, string> = {
    completed: 'text-[var(--color-success)]',
    'in-progress': 'text-[var(--color-accent)]',
    available: 'text-[var(--color-primary)]',
    locked: 'text-[var(--text-muted)]',
  }

  return (
    <motion.aside
      className={clsx(
        'flex flex-col h-screen bg-[var(--bg-surface)] border-r border-[var(--border-default)] z-[150] transition-colors duration-300',
        isMobile
          ? 'fixed top-0 left-0 w-72 shadow-2xl'
          : 'relative w-64 min-w-[256px]'
      )}
      initial={{ x: isMobile ? -288 : 0 }}
      animate={{ x: 0 }}
      exit={{ x: -288 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-5 pb-4">
        <div className="flex items-center gap-3 cursor-pointer select-none" onClick={() => handleNav('/')}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white">
            <Brain size={20} />
          </div>
          <span className="font-mono text-lg font-extrabold tracking-widest bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            EPOCH
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all"
            title={theme === 'dark' ? 'Mode clair' : 'Mode sombre'}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          {isMobile && (
            <button onClick={onClose} className="p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all">
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 pb-2 space-y-0.5">
        <button
          className={clsx(
            'flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 text-left',
            location.pathname === '/'
              ? 'bg-primary/10 text-primary'
              : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]'
          )}
          onClick={() => handleNav('/')}
        >
          <Map size={18} />
          <span>Roadmap</span>
        </button>

        <div className="h-px bg-[var(--border-subtle)] my-3 mx-1" />

        <span className="block text-[10px] font-bold tracking-[2px] uppercase text-[var(--text-muted)] px-3 pb-2">
          Chapitres
        </span>

        {courses.map(course => {
          const isActive = location.pathname === `/lab/${course.id}`
          const isLocked = course.status === 'locked'
          return (
            <button
              key={course.id}
              className={clsx(
                'flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-[13px] font-medium transition-all duration-200 text-left relative',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : isLocked
                  ? 'text-[var(--text-muted)] opacity-40 cursor-not-allowed'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]'
              )}
              onClick={() => !isLocked && handleNav(`/lab/${course.id}`)}
              disabled={isLocked}
              title={isLocked ? 'Complétez les prérequis' : course.title}
            >
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r bg-primary" />
              )}
              <Zap size={14} className={statusColors[course.status] || ''} />
              <span className="truncate">{course.shortTitle}</span>
              {course.status === 'completed' && (
                <span className="ml-auto text-[var(--color-success)] text-xs font-bold">✓</span>
              )}
            </button>
          )
        })}
      </nav>

      {/* Progress */}
      <div className="px-5 py-4 border-t border-[var(--border-default)]">
        <div className="text-[10px] font-semibold tracking-widest uppercase text-[var(--text-muted)] mb-2">
          Progression
        </div>
        <div className="h-1.5 bg-[var(--bg-overlay)] rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>
        <div className="text-right font-mono text-xs font-semibold text-primary mt-1.5">
          {progress}%
        </div>
      </div>
    </motion.aside>
  )
}
