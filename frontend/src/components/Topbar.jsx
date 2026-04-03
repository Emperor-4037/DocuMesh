import { Sparkles, Zap } from 'lucide-react';

export default function Topbar() {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="topbar-brand-icon">
          <Sparkles size={16} />
        </div>
        <span>AI Writing Assistant</span>
        <span className="topbar-badge">Beta</span>
      </div>
      <div className="topbar-right">
        <div className="flex items-center gap-2 text-sm text-muted">
          <div className="status-dot" />
          <span style={{ color: 'var(--clr-subtle)', fontSize: '0.8rem' }}>Services online</span>
        </div>
        <div className="flex items-center gap-2 badge badge-purple">
          <Zap size={12} />
          6 microservices
        </div>
      </div>
    </header>
  );
}
