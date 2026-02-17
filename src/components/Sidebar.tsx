import { useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Map, Zap, X } from 'lucide-react'
import type { CourseNode } from '../data/courses'
import { getTotalProgress } from '../data/courses'
import styles from './Sidebar.module.css'

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

  const handleNav = (path: string) => {
    navigate(path)
    if (isMobile && onClose) onClose()
  }

  // On mobile, hide when closed
  if (isMobile && !isOpen) return null

  return (
    <motion.aside
      className={`${styles.sidebar} ${isMobile ? styles.mobileDrawer : ''}`}
      initial={{ x: isMobile ? -280 : -80 }}
      animate={{ x: 0 }}
      exit={{ x: -280 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      {/* Mobile close button */}
      {isMobile && (
        <button className={styles.closeBtn} onClick={onClose} aria-label="Close menu">
          <X size={20} />
        </button>
      )}

      {/* Logo */}
      <div className={styles.logo} onClick={() => handleNav('/')}>
        <div className={styles.logoIcon}>
          <Brain size={22} />
        </div>
        <span className={styles.logoText}>EPOCH</span>
      </div>

      {/* Nav */}
      <nav className={styles.nav}>
        <button
          className={`${styles.navItem} ${location.pathname === '/' ? styles.active : ''}`}
          onClick={() => handleNav('/')}
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
              onClick={() => !isLocked && handleNav(`/lab/${course.id}`)}
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
