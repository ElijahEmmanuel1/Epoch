import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Play, Lock, CheckCircle, Circle } from 'lucide-react'
import { clsx } from 'clsx'
import type { CourseNode } from '../data/courses'
import { nodePositions, getTotalProgress } from '../data/courses'

interface Props {
  courses: CourseNode[]
  updateCourse: (id: string, partial: Partial<CourseNode>) => void
}

export default function Roadmap({ courses, updateCourse: _updateCourse }: Props) {
  const navigate = useNavigate()
  const progress = getTotalProgress(courses)

  const edges: { from: string; to: string }[] = []
  courses.forEach(c => {
    c.dependencies.forEach(dep => {
      edges.push({ from: dep, to: c.id })
    })
  })

  const getStatusColor = (status: CourseNode['status']) => {
    switch (status) {
      case 'completed': return 'var(--color-success)'
      case 'in-progress': return 'var(--color-accent)'
      case 'available': return 'var(--color-primary)'
      case 'locked': return 'var(--text-muted)'
    }
  }

  const getStatusIcon = (status: CourseNode['status']) => {
    switch (status) {
      case 'completed': return <CheckCircle size={16} />
      case 'in-progress': return <Play size={16} />
      case 'available': return <Circle size={16} />
      case 'locked': return <Lock size={14} />
    }
  }

  return (
    <div className="flex flex-col h-screen bg-[var(--bg-base)] overflow-hidden transition-colors duration-300">
      {/* Header */}
      <motion.header
        className="flex items-center justify-between px-6 md:px-8 py-5 border-b border-[var(--border-default)] bg-[var(--bg-surface)] shrink-0 transition-colors"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col gap-1 pl-12 lg:pl-0">
          <h1 className="text-xl md:text-2xl font-extrabold tracking-tight bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Neural Roadmap
          </h1>
          <p className="text-xs md:text-sm text-[var(--text-muted)]">
            Votre parcours d'apprentissage Deep Learning
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative w-12 h-12 md:w-14 md:h-14">
            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
              <circle cx="50" cy="50" r="42" fill="none" stroke="var(--bg-overlay)" strokeWidth="6" />
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke="url(#progressGrad)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={`${progress * 2.64} ${264 - progress * 2.64}`}
                strokeDashoffset="66"
                className="transition-all duration-1000"
              />
              <defs>
                <linearGradient id="progressGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--color-primary)" />
                  <stop offset="100%" stopColor="var(--color-accent)" />
                </linearGradient>
              </defs>
            </svg>
            <span className="absolute inset-0 flex items-center justify-center font-mono text-xs md:text-sm font-bold text-primary">
              {progress}%
            </span>
          </div>
        </div>
      </motion.header>

      {/* Graph */}
      <div className="flex-1 relative overflow-auto p-6 md:p-10" style={{ minHeight: 600 }}>
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
          {edges.map((edge, i) => {
            const from = nodePositions[edge.from]
            const to = nodePositions[edge.to]
            if (!from || !to) return null
            const fromNode = courses.find(c => c.id === edge.from)
            const isActive = fromNode?.status === 'completed'
            return (
              <motion.line
                key={i}
                x1={from.x + 90} y1={from.y + 50}
                x2={to.x + 90} y2={to.y + 50}
                stroke={isActive ? 'var(--color-primary)' : 'var(--border-default)'}
                strokeWidth={isActive ? 2 : 1}
                strokeOpacity={isActive ? 0.5 : 0.3}
                strokeDasharray={isActive ? 'none' : '6 4'}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1, delay: i * 0.1 }}
              />
            )
          })}
        </svg>

        {courses.map((course, i) => {
          const pos = nodePositions[course.id]
          if (!pos) return null
          const isClickable = course.status !== 'locked'
          return (
            <motion.div
              key={course.id}
              className={clsx(
                'absolute w-44 md:w-48 rounded-xl border overflow-hidden z-[1] transition-all duration-300',
                'bg-[var(--bg-surface)] border-[var(--border-default)]',
                isClickable ? 'cursor-pointer hover:shadow-lg hover:border-[var(--border-active)]' : 'cursor-not-allowed opacity-50',
                course.status === 'completed' && 'border-[var(--color-success)]/30',
                course.status === 'in-progress' && 'border-[var(--color-accent)]/40'
              )}
              style={{ left: pos.x, top: pos.y }}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: i * 0.06 }}
              whileHover={isClickable ? { scale: 1.04, y: -3 } : {}}
              onClick={() => isClickable && navigate(`/lab/${course.id}`)}
            >
              <div className="h-1 w-full" style={{ background: getStatusColor(course.status) }} />
              <div className="p-3.5">
                <div className="flex items-center justify-between mb-2">
                  <span style={{ color: getStatusColor(course.status) }}>
                    {getStatusIcon(course.status)}
                  </span>
                  <span className="text-[9px] font-bold tracking-widest uppercase text-[var(--text-muted)] bg-[var(--bg-overlay)] px-1.5 py-0.5 rounded">
                    {course.category}
                  </span>
                </div>
                <h3 className="text-[13px] md:text-sm font-bold text-[var(--text-primary)] mb-1 leading-tight">
                  {course.shortTitle}
                </h3>
                <p className="text-[10px] md:text-[11px] text-[var(--text-secondary)] leading-relaxed line-clamp-2 mb-2">
                  {course.description.slice(0, 70)}…
                </p>
                {course.status !== 'locked' && (
                  <div className="h-1 bg-[var(--bg-overlay)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${course.progress}%`, background: getStatusColor(course.status) }}
                    />
                  </div>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
