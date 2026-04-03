import { Wand2, ShieldCheck, BookOpen, Mic2, AlignLeft, Database, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import PageWrapper from '../components/PageWrapper';

const TOOLS = [
  {
    id: 'paraphrase',
    icon: Wand2,
    label: 'Paraphrase',
    desc: 'Rewrite text in 5 different tones while preserving meaning.',
    color: '#a78bfa',
    bg: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(167,139,250,0.2))',
  },
  {
    id: 'grammar',
    icon: ShieldCheck,
    label: 'Grammar Fix',
    desc: 'Detect and correct grammatical errors with detailed diffs.',
    color: '#34d399',
    bg: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(52,211,153,0.2))',
  },
  {
    id: 'simplify',
    icon: BookOpen,
    label: 'Simplify',
    desc: 'Adapt content to 5 reading levels for wider accessibility.',
    color: '#60a5fa',
    bg: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(96,165,250,0.2))',
  },
  {
    id: 'tone',
    icon: Mic2,
    label: 'Tone Adjust',
    desc: 'Transform writing between 6 professional and personal styles.',
    color: '#fbbf24',
    bg: 'linear-gradient(135deg, rgba(245,158,11,0.1), rgba(251,191,36,0.2))',
  },
  {
    id: 'summarize',
    icon: AlignLeft,
    label: 'Summarize',
    desc: 'Condense long documents with abstractive BART summarization.',
    color: '#f87171',
    bg: 'linear-gradient(135deg, rgba(239,68,68,0.1), rgba(248,113,113,0.2))',
  },
  {
    id: 'rag',
    icon: Database,
    label: 'Doc Assistant',
    desc: 'Upload PDFs and chat with your documents using RAG + LLM.',
    color: '#c084fc',
    bg: 'linear-gradient(135deg, rgba(168,85,247,0.1), rgba(192,132,252,0.2))',
  },
];

const STATS = [
  { label: 'Microservices', value: '6' },
  { label: 'Vector DB', value: 'Qdrant' },
  { label: 'LLM Model', value: 'TinyLlama' },
  { label: 'Queue', value: 'Celery' },
  { label: 'Metrics', value: 'Prometheus' },
  { label: 'Tracing', value: 'OTLP' },
];

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
};

export default function DashboardPage({ setActive }) {
  return (
    <PageWrapper>
      <div className="page-header">
        <motion.h1 initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>AI Writing Assistant</motion.h1>
        <motion.p initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          A production-grade microservices platform for AI-powered text processing and document intelligence.
        </motion.p>
      </div>

      {/* Stats */}
      <motion.div className="stats-row mb-3 mt-4" variants={containerVariants} initial="hidden" animate="show">
        {STATS.map((s) => (
          <motion.div key={s.label} className="stat-card" variants={itemVariants}>
            <div className="stat-card-label">{s.label}</div>
            <div className="stat-card-value">{s.value}</div>
          </motion.div>
        ))}
      </motion.div>

      {/* Divider */}
      <div className="divider" style={{ margin: '32px 0 16px' }} />

      <div style={{ margin: '0 0 20px', fontWeight: 800, color: 'var(--clr-text)', fontSize: '1.1rem', letterSpacing: '0.02em', fontFamily: 'var(--font-heading)' }}>
        Explore Available Tools
      </div>

      <motion.div className="tool-grid" variants={containerVariants} initial="hidden" animate="show">
        {TOOLS.map((tool) => {
          const Icon = tool.icon;
          return (
            <motion.div
              key={tool.id}
              className="tool-card"
              variants={itemVariants}
              onClick={() => setActive(tool.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="tool-card-icon" style={{ background: tool.bg, color: tool.color }}>
                <Icon size={24} strokeWidth={2.5} />
              </div>
              <h3>{tool.label}</h3>
              <p>{tool.desc}</p>
              <div className="flex items-center gap-2 mt-4" style={{ color: tool.color, fontSize: '0.9rem', fontWeight: 700 }}>
                Open tool <ArrowRight size={16} />
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Architecture note */}
      <motion.div 
        className="glass-panel mt-4" 
        style={{ borderColor: 'rgba(139,92,246,0.3)', marginTop: '40px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <Database size={20} style={{ color: '#c4b5fd' }} />
          <span style={{ fontWeight: 800, fontSize: '1rem', color: '#fff', fontFamily: 'var(--font-heading)' }}>Platform Architecture</span>
        </div>
        <p className="text-sm text-muted" style={{ lineHeight: 1.8 }}>
          All requests flow through the <strong style={{ color: '#d8b4fe' }}>FastAPI Gateway</strong> → individual microservices.
          Document ingestion is handled asynchronously via <strong style={{ color: '#d8b4fe' }}>Celery + Redis</strong>.
          Vectors are stored in <strong style={{ color: '#d8b4fe' }}>Qdrant</strong> and retrieved with hybrid search + cross-encoder reranking before being passed to <strong style={{ color: '#d8b4fe' }}>TinyLlama</strong>.
        </p>
      </motion.div>
    </PageWrapper>
  );
}
