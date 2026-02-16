import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Play, Lock, CheckCircle, Circle } from 'lucide-react'
import type { CourseNode } from '../data/courses'
import { nodePositions, getTotalProgress } from '../data/courses'
import styles from './Roadmap.module.css'

interface Props {
  courses: CourseNode[]
  updateCourse: (id: string, partial: Partial<CourseNode>) => void
}

export default function Roadmap({ courses, updateCourse: _updateCourse }: Props) {
  const navigate = useNavigate()
  const progress = getTotalProgress(courses)

  // Build edges from dependencies
  const edges: { from: string; to: string }[] = []
  courses.forEach(c => {
    c.dependencies.forEach(dep => {
      edges.push({ from: dep, to: c.id })
    })
  })

  const getStatusColor = (status: CourseNode['status']) => {
    switch (status) {
      case 'completed': return 'var(--green)'
      case 'in-progress': return 'var(--cyan)'
      case 'available': return 'var(--magenta)'
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
    <div className={styles.container}>
      {/* Header */}
      <motion.header
        className={styles.header}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>Neural Roadmap</h1>
          <p className={styles.subtitle}>Votre parcours d'apprentissage Deep Learning</p>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.progressCircle}>
            <svg viewBox="0 0 100 100">
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke="var(--bg-surface)"
                strokeWidth="6"
              />
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke="url(#progressGrad)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={`${progress * 2.64} ${264 - progress * 2.64}`}
                strokeDashoffset="66"
                style={{ transition: 'stroke-dasharray 1s ease' }}
              />
              <defs>
                <linearGradient id="progressGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--cyan)" />
                  <stop offset="100%" stopColor="var(--magenta)" />
                </linearGradient>
              </defs>
            </svg>
            <span className={styles.progressText}>{progress}%</span>
          </div>
        </div>
      </motion.header>

      {/* Graph */}
      <div className={styles.graph}>
        {/* SVG Edges */}
        <svg className={styles.edgesSvg}>
          {edges.map((edge, i) => {
            const from = nodePositions[edge.from]
            const to = nodePositions[edge.to]
            if (!from || !to) return null
            const fromNode = courses.find(c => c.id === edge.from)
            const isActive = fromNode?.status === 'completed'
            return (
              <motion.line
                key={i}
                x1={from.x + 80}
                y1={from.y + 40}
                x2={to.x + 80}
                y2={to.y + 40}
                stroke={isActive ? 'var(--cyan-dim)' : 'var(--border)'}
                strokeWidth={isActive ? 2 : 1}
                strokeDasharray={isActive ? 'none' : '6 4'}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1, delay: i * 0.1 }}
              />
            )
          })}
        </svg>

        {/* Nodes */}
        {courses.map((course, i) => {
          const pos = nodePositions[course.id]
          if (!pos) return null
          const isClickable = course.status !== 'locked'
          return (
            <motion.div
              key={course.id}
              className={`${styles.node} ${styles[course.status]}`}
              style={{ left: pos.x, top: pos.y }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: i * 0.06 }}
              whileHover={isClickable ? { scale: 1.05, y: -4 } : {}}
              onClick={() => isClickable && navigate(`/lab/${course.id}`)}
            >
              <div
                className={styles.nodeGlow}
                style={{ background: getStatusColor(course.status) }}
              />
              <div className={styles.nodeContent}>
                <div className={styles.nodeHeader}>
                  <span className={styles.nodeStatus} style={{ color: getStatusColor(course.status) }}>
                    {getStatusIcon(course.status)}
                  </span>
                  <span className={styles.nodeCategory}>{course.category}</span>
                </div>
                <h3 className={styles.nodeTitle}>{course.shortTitle}</h3>
                <p className={styles.nodeDesc}>{course.description.slice(0, 60)}…</p>
                {course.status !== 'locked' && (
                  <div className={styles.nodeProgress}>
                    <div className={styles.nodeProgressBar}>
                      <div
                        className={styles.nodeProgressFill}
                        style={{
                          width: `${course.progress}%`,
                          background: getStatusColor(course.status),
                        }}
                      />
                    </div>
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
