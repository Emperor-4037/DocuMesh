import { useState } from 'react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wand2, ShieldCheck, BookOpen, Mic2, AlignLeft,
  Database, LayoutDashboard
} from 'lucide-react';

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, section: 'Overview' },
  { id: 'paraphrase', label: 'Paraphrase',   icon: Wand2,       section: 'NLP Tools' },
  { id: 'grammar',    label: 'Grammar Fix',  icon: ShieldCheck, section: 'NLP Tools' },
  { id: 'simplify',   label: 'Simplify',     icon: BookOpen,    section: 'NLP Tools' },
  { id: 'tone',       label: 'Tone Adjust',  icon: Mic2,        section: 'NLP Tools' },
  { id: 'summarize',  label: 'Summarize',    icon: AlignLeft,   section: 'NLP Tools' },
  { id: 'rag',        label: 'Doc Assistant',icon: Database,    section: 'RAG' },
];

const SECTIONS = ['Overview', 'NLP Tools', 'RAG'];

export default function Sidebar({ active, setActive }) {
  return (
    <aside className="sidebar">
      {SECTIONS.map((section) => (
        <div key={section}>
          <div className="sidebar-section-label">{section}</div>
          <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 4 }}>
            {NAV.filter((n) => n.section === section).map((item) => {
              const Icon = item.icon;
              const isActive = active === item.id;
              
              return (
                <button
                  key={item.id}
                  id={`nav-${item.id}`}
                  className={clsx('nav-item', { active: isActive })}
                  onClick={() => setActive(item.id)}
                  style={{ position: 'relative', background: 'transparent', border: 'none', width: '100%' }}
                >
                  <Icon size={18} className="nav-item-icon" style={{ zIndex: 2, position: 'relative', color: isActive ? '#fff' : 'inherit' }} />
                  <span style={{ zIndex: 2, position: 'relative', color: isActive ? '#fff' : 'inherit' }}>{item.label}</span>
                  
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active-bg"
                      className="active-bg"
                      initial={false}
                      transition={{ type: 'spring', stiffness: 300, damping: 24 }}
                      style={{
                        position: 'absolute',
                        inset: 0,
                        background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(99,102,241,0.15))',
                        borderRadius: '16px',
                        border: '1px solid rgba(139, 92, 246, 0.5)',
                        boxShadow: '0 4px 15px rgba(124, 58, 237, 0.2)',
                        zIndex: 0
                      }}
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </aside>
  );
}
