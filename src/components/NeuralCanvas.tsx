import { useState } from 'react'
import { motion } from 'framer-motion'
import 'katex/dist/katex.min.css'
import { BlockMath } from 'react-katex'
import { Lightbulb, AlertTriangle, Info, HelpCircle } from 'lucide-react'
import type { CourseNode, Exercise } from '../data/courses'
import NeuralDiagram from './NeuralDiagram'
import styles from './NeuralCanvas.module.css'

interface Props {
  course: CourseNode
  activeTab: 'theory' | 'exercise'
  currentExercise?: Exercise
  highlightedVar: string | null
}

export default function NeuralCanvas({ course, activeTab, currentExercise, highlightedVar }: Props) {
  return (
    <div className={styles.canvas}>
      {activeTab === 'theory' ? (
        <TheoryView
          theory={course.theory}
          highlightedVar={highlightedVar}
          courseId={course.id}
        />
      ) : currentExercise ? (
        <ExerciseView exercise={currentExercise} />
      ) : (
        <div className={styles.empty}>
          <Info size={24} />
          <p>Aucun exercice pour ce module</p>
        </div>
      )}
    </div>
  )
}

function TheoryView({
  theory,
  highlightedVar,
  courseId,
}: {
  theory: TheoryBlock[]
  highlightedVar: string | null
  courseId: string
}) {
  return (
    <div className={styles.theory}>
      {/* Neural Network Diagram */}
      <div className={styles.diagramSection}>
        <NeuralDiagram courseId={courseId} highlightedVar={highlightedVar} />
      </div>

      {/* Theory Blocks */}
      {theory.map((block, i) => (
        <motion.div
          key={i}
          className={styles.block}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06 }}
        >
          {block.type === 'text' && (
            <div
              className={styles.text}
              dangerouslySetInnerHTML={{ __html: formatMarkdown(block.content) }}
            />
          )}

          {block.type === 'equation' && (
            <div
              className={`${styles.equation} ${
                highlightedVar && block.highlightVar === highlightedVar
                  ? styles.equationHighlight
                  : ''
              }`}
            >
              {block.label && <span className={styles.eqLabel}>{block.label}</span>}
              <div className={styles.eqContent}>
                <BlockMath math={block.content} />
              </div>
            </div>
          )}

          {block.type === 'callout' && (
            <div className={styles.callout}>
              <div className={styles.calloutIcon}>
                {block.content.startsWith('💡') ? <Lightbulb size={16} /> :
                 block.content.startsWith('⚠') ? <AlertTriangle size={16} /> :
                 block.content.startsWith('🧠') ? <HelpCircle size={16} /> :
                 <Info size={16} />}
              </div>
              <div
                className={styles.calloutText}
                dangerouslySetInnerHTML={{
                  __html: formatMarkdown(block.content.replace(/^[💡⚠🧠⚡]\s*/, '')),
                }}
              />
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}

function ExerciseView({ exercise }: { exercise: Exercise }) {
  const [showHints, setShowHints] = useState(false)
  return (
    <div className={styles.exercise}>
      <div className={styles.exerciseHeader}>
        <h3 className={styles.exerciseTitle}>{exercise.title}</h3>
      </div>
      <div
        className={styles.exerciseInstructions}
        dangerouslySetInnerHTML={{ __html: formatMarkdown(exercise.instructions) }}
      />
      {exercise.hints.length > 0 && (
        <div className={styles.hints}>
          <button
            className={styles.hintToggle}
            onClick={() => setShowHints(!showHints)}
          >
            <Lightbulb size={14} />
            {showHints ? 'Masquer les indices' : `${exercise.hints.length} indice(s) disponible(s)`}
          </button>
          {showHints && (
            <motion.ul
              className={styles.hintList}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
            >
              {exercise.hints.map((hint, i) => (
                <li key={i} className={styles.hintItem}>{hint}</li>
              ))}
            </motion.ul>
          )}
        </div>
      )}
    </div>
  )
}

// Simple Markdown formatter
function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>')
}

