import { Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Roadmap from './pages/Roadmap'
import Lab from './pages/Lab'
import { courseNodes as initialCourses, type CourseNode } from './data/courses'

export default function App() {
  const [courses, setCourses] = useState<CourseNode[]>(initialCourses)

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
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar courses={courses} />
      <main style={{ flex: 1, overflow: 'hidden' }}>
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
