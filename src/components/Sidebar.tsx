import { useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Map, Zap } from 'lucide-react'
import type { CourseNode } from '../data/courses'
import { getTotalProgress } from '../data/courses'
import styles from './Sidebar.module.css'

interface Props {
  courses: CourseNode[]
}

export default function Sidebar({ courses }: Props) {
  const navigate = useNavigate()
  const location = useLocation()
  const progress = getTotalProgress(courses)

  return (
    <motion.aside
      className={styles.sidebar}
      initial={{ x: -80 }}
      animate={{ x: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      {/* Logo */}
      <div className={styles.logo} onClick={() => navigate('/')}>
        <div className={styles.logoIcon}>
          <Brain size={22} />
        </div>
        <span className={styles.logoText}>EPOCH</span>
      </div>

      {/* Nav */}
      <nav className={styles.nav}>
        <button
          className={`${styles.navItem} ${location.pathname === '/' ? styles.active : ''}`}
          onClick={() => navigate('/')}
        >
          <Map size={18} />
          <span>Roadmap</span>
        </button>

        <div className={styles.divider} />
        <span className={styles.sectionLabel}>Modules</span>

        {courses.map(course => {
          const isActive = location.pathname === `/lab/${course.id}`
          const isLocked = course.status === 'locked'
          return (
            <button
              key={course.id}
              className={`${styles.navItem} ${isActive ? styles.active : ''} ${isLocked ? styles.locked : ''}`}
              onClick={() => !isLocked && navigate(`/lab/${course.id}`)}
              disabled={isLocked}
              title={isLocked ? 'Complétez les prérequis' : course.title}
            >
              <Zap size={14} className={styles.nodeIcon} data-status={course.status} />
              <span>{course.shortTitle}</span>
              {course.status === 'completed' && <span className={styles.check}>✓</span>}
            </button>
          )
        })}
      </nav>

      {/* Progress */}
      <div className={styles.progress}>
        <div className={styles.progressLabel}>Training Progress</div>
        <div className={styles.progressBar}>
          <motion.div
            className={styles.progressFill}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>
        <div className={styles.progressValue}>{progress}%</div>
      </div>
    </motion.aside>
  )
}
