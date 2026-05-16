import { useState, useEffect, useCallback } from 'react';
import { Activity, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

// Use empty string so requests go through Vite's dev proxy
const DEMO_TOKEN = import.meta.env.VITE_DEMO_TOKEN ?? 'demo-token';

const SERVICES = [
  { id: 'gateway', label: 'Gateway' },
  { id: 'paraphrase', label: 'Paraphrase' },
  { id: 'grammar', label: 'Grammar' },
  { id: 'simplify', label: 'Simplify' },
  { id: 'tone', label: 'Tone' },
  { id: 'summarize', label: 'Summarize' },
  { id: 'rag', label: 'RAG' },
];

export default function HealthDashboard() {
  const [statuses, setStatuses] = useState({});
  const [checking, setChecking] = useState(false);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    const results = {};

    // Check gateway health first (proxied through Vite dev server)
    try {
      const start = Date.now();
      const resp = await fetch('/health', {
        headers: { Authorization: `Bearer ${DEMO_TOKEN}` },
        signal: AbortSignal.timeout(5000),
      });
      const latency = Date.now() - start;
      results['gateway'] = { ok: resp.ok, latency, status: resp.status };
    } catch {
      results['gateway'] = { ok: false, latency: null, error: 'unreachable' };
    }

    // Check all downstream services via the gateway's /readiness endpoint
    try {
      const start = Date.now();
      const resp = await fetch('/readiness', {
        headers: { Authorization: `Bearer ${DEMO_TOKEN}` },
        signal: AbortSignal.timeout(8000),
      });
      const latency = Date.now() - start;
      if (resp.ok || resp.status === 503) {
        const data = await resp.json();
        const svcMap = data.services ?? {};
        for (const svc of ['paraphrase', 'grammar', 'simplify', 'tone', 'summarize', 'rag']) {
          const info = svcMap[svc];
          if (info) {
            results[svc] = { ok: info.status === 'ok', latency, status: info.code };
          } else {
            results[svc] = { ok: false, latency: null, error: 'no data' };
          }
        }
      }
    } catch {
      // If readiness check fails, mark all downstream as unknown
      for (const svc of ['paraphrase', 'grammar', 'simplify', 'tone', 'summarize', 'rag']) {
        results[svc] = results[svc] ?? { ok: false, latency: null, error: 'unreachable' };
      }
    }

    setStatuses(results);
    setChecking(false);
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const allOk = Object.values(statuses).every(s => s.ok);

  return (
    <motion.div
      className="glass-panel"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <div className="flex items-center gap-3 mb-3">
        <Activity size={18} style={{ color: allOk ? 'var(--clr-success)' : '#f87171' }} />
        <span style={{ fontWeight: 800, fontSize: '1rem', color: 'var(--clr-text)', fontFamily: 'var(--font-heading)' }}>
          Service Health
        </span>
        <button
          className="btn btn-ghost btn-icon btn-sm"
          onClick={checkHealth}
          disabled={checking}
          style={{ marginLeft: 'auto' }}
          title="Refresh health status"
        >
          <RefreshCw size={14} className={checking ? 'spin' : ''} />
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 8 }}>
        {SERVICES.map(svc => {
          const s = statuses[svc.id];
          const ok = s?.ok;
          const latency = s?.latency;
          return (
            <div
              key={svc.id}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px',
                borderRadius: 8,
                background: ok ? 'rgba(52,211,153,0.08)' : s ? 'rgba(248,113,113,0.08)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${ok ? 'rgba(52,211,153,0.2)' : s ? 'rgba(248,113,113,0.2)' : 'rgba(255,255,255,0.06)'}`,
              }}
            >
              {ok ? (
                <CheckCircle2 size={14} style={{ color: 'var(--clr-success)', flexShrink: 0 }} />
              ) : s ? (
                <XCircle size={14} style={{ color: '#f87171', flexShrink: 0 }} />
              ) : (
                <div style={{ width: 14, height: 14, borderRadius: '50%', background: 'rgba(255,255,255,0.1)', flexShrink: 0 }} />
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--clr-text)' }}>{svc.label}</div>
                {latency != null && (
                  <div style={{ fontSize: '0.65rem', color: 'var(--clr-muted)' }}>{latency}ms</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
