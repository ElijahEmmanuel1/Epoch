import { useState } from 'react'
import { motion } from 'framer-motion'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'
import { Lightbulb, AlertTriangle, Info, HelpCircle } from 'lucide-react'
import type { CourseNode, Exercise, TheoryBlock } from '../data/courses'
import NeuralDiagram from './NeuralDiagram'
import DiagramRenderer, { hasDiagram } from './diagrams/DiagramRegistry'
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



// ... imports remain the same ...

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
            <div className={styles.text}>
              <MarkdownRenderer content={block.content} />
            </div>
          )}

          {block.type === 'equation' && (
            <div
              className={`${styles.equation} ${highlightedVar && block.highlightVar === highlightedVar
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

          {block.type === 'diagram' && (
            <div className={styles.inlineDiagram}>
              {block.label && <span className={styles.diagramLabel}>{block.label}</span>}
              {block.diagramId && hasDiagram(block.diagramId) ? (
                <DiagramRenderer diagramId={block.diagramId} />
              ) : (
                <pre className={styles.diagramPre}>{block.content}</pre>
              )}
            </div>
          )}

          {block.type === 'code' && (
            <div className={styles.codeBlock}>
              {block.title && <div className={styles.codeTitle}>{block.title}</div>}
              <pre>
                <code className={`language-${block.language || 'python'}`}>
                  {block.content}
                </code>
              </pre>
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
              <div className={styles.calloutText}>
                <MarkdownRenderer content={block.content.replace(/^[💡⚠🧠⚡]\s*/, '')} />
              </div>
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}

// ... ExerciseView ...

// ── Markdown + LaTeX Parser ──
function MarkdownRenderer({ content }: { content: string }) {
  // 1. Split by LaTeX delimiters ($...$)
  const parts = content.split(/(\$[^$]+\$)/g)

  return (
    <span>
      {parts.map((part, index) => {
        if (part.startsWith('$') && part.endsWith('$')) {
          // Render inline math
          return <InlineMath key={index} math={part.slice(1, -1)} />
        }
        // Render markdown text
        return <span key={index} dangerouslySetInnerHTML={{ __html: formatMarkdown(part) }} />
      })}
    </span>
  )
}

function formatMarkdown(text: string): string {
  // Escape HTML first to prevent XSS (basic)
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")

  return safe
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>')
}

