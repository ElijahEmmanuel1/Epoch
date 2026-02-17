import { Routes, Route } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { Menu } from 'lucide-react'
import Sidebar from './components/Sidebar'
import Roadmap from './pages/Roadmap'
import Lab from './pages/Lab'
import { courseNodes as initialCourses, type CourseNode } from './data/courses'

function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(window.innerWidth < breakpoint)
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < breakpoint)
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [breakpoint])
  return isMobile
}

export default function App() {
  const [courses, setCourses] = useState<CourseNode[]>(initialCourses)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const isMobile = useIsMobile()
  const location = useLocation()

  // Close sidebar on route change (mobile)
  useEffect(() => {
    if (isMobile) setSidebarOpen(false)
  }, [location.pathname, isMobile])

  const updateCourse = (id: string, partial: Partial<CourseNode>) => {
    setCourses(prev => {
      // 1. Update the target course
      const newCourses = prev.map(c => (c.id === id ? { ...c, ...partial } : c))

      // 2. Check for newly unlocked courses
      // We only need to check if the status changed to 'completed'
      if (partial.status === 'completed') {
        let changed = true
        // Propagate changes until stable (to handle chains if multiple unlock at once, though usually one-by-one)
        while (changed) {
          changed = false
          newCourses.forEach(course => {
            if (course.status === 'locked') {
              // Check if all dependencies are completed
              const allDependenciesMet = course.dependencies.every(depId => {
                const dep = newCourses.find(c => c.id === depId)
                return dep?.status === 'completed'
              })

              if (allDependenciesMet) {
                course.status = 'available'
                changed = true
              }
            }
          })
        }
      }
      
      return newCourses
    })
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', position: 'relative' }}>
      {/* Mobile hamburger */}
      {isMobile && !sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          aria-label="Open menu"
          style={{
            position: 'fixed',
            top: 12,
            left: 12,
            zIndex: 201,
            width: 44,
            height: 44,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--cyan)',
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
        >
          <Menu size={22} />
        </button>
      )}

      {/* Overlay */}
      {isMobile && sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.6)',
            zIndex: 149,
            backdropFilter: 'blur(2px)',
          }}
        />
      )}

      <Sidebar
        courses={courses}
        isMobile={isMobile}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main style={{ flex: 1, overflow: 'hidden', minWidth: 0 }}>
        <Routes>
          <Route
            path="/"
            element={<Roadmap courses={courses} updateCourse={updateCourse} />}
          />
          <Route
            path="/lab/:courseId"
            element={<Lab courses={courses} updateCourse={updateCourse} />}
          />
        </Routes>
      </main>
    </div>
  )
}
