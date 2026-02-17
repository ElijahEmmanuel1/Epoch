import { Routes, Route, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { Menu } from 'lucide-react'
import Sidebar from './components/Sidebar'
import Roadmap from './pages/Roadmap'
import Lab from './pages/Lab'
import { courseNodes as initialCourses, type CourseNode } from './data/courses'

function useIsMobile(bp = 1024) {
  const [m, setM] = useState(window.innerWidth < bp)
  useEffect(() => {
    const h = () => setM(window.innerWidth < bp)
    window.addEventListener('resize', h)
    return () => window.removeEventListener('resize', h)
  }, [bp])
  return m
}

export default function App() {
  const [courses, setCourses] = useState<CourseNode[]>(initialCourses)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const isMobile = useIsMobile()
  const location = useLocation()

  useEffect(() => {
    if (isMobile) setSidebarOpen(false)
  }, [location.pathname, isMobile])

  const updateCourse = (id: string, partial: Partial<CourseNode>) => {
    setCourses(prev => {
      const newCourses = prev.map(c => (c.id === id ? { ...c, ...partial } : c))
      if (partial.status === 'completed') {
        let changed = true
        while (changed) {
          changed = false
          newCourses.forEach(course => {
            if (course.status === 'locked') {
              const allDeps = course.dependencies.every(depId => {
                const dep = newCourses.find(c => c.id === depId)
                return dep?.status === 'completed'
              })
              if (allDeps) { course.status = 'available'; changed = true }
            }
          })
        }
      }
      return newCourses
    })
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-base)] transition-colors duration-300">
      {isMobile && !sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          aria-label="Open menu"
          className="fixed top-3 left-3 z-[201] flex items-center justify-center w-11 h-11 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-default)] text-[var(--text-secondary)] hover:text-primary hover:border-[var(--border-active)] transition-all duration-200 backdrop-blur-sm"
        >
          <Menu size={20} />
        </button>
      )}

      {isMobile && sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-[149] bg-black/50 backdrop-blur-sm"
        />
      )}

      <Sidebar
        courses={courses}
        isMobile={isMobile}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex-1 overflow-hidden min-w-0">
        <Routes>
          <Route path="/" element={<Roadmap courses={courses} updateCourse={updateCourse} />} />
          <Route path="/lab/:courseId" element={<Lab courses={courses} updateCourse={updateCourse} />} />
        </Routes>
      </main>
    </div>
  )
}
