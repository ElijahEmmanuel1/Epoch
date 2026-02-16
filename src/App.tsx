import { Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Roadmap from './pages/Roadmap'
import Lab from './pages/Lab'
import { courseNodes as initialCourses, type CourseNode } from './data/courses'

export default function App() {
  const [courses, setCourses] = useState<CourseNode[]>(initialCourses)

  const updateCourse = (id: string, partial: Partial<CourseNode>) => {
    setCourses(prev =>
      prev.map(c => (c.id === id ? { ...c, ...partial } : c))
    )
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
