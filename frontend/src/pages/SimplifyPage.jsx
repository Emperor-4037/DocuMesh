import { useState } from 'react';
import PageWrapper from '../components/PageWrapper';
import { BookOpen, ArrowRight } from 'lucide-react';
import { simplify } from '../api';
import { useToast } from '../context/ToastContext';
import { ResultBox, Spinner, FormGroup } from '../components/ui';

const LEVELS = ['elementary', 'middle school', 'high school', 'college', 'expert'];

export default function SimplifyPage() {
  const [text, setText] = useState('');
  const [level, setLevel] = useState('middle school');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult('');
    try {
      const { data } = await simplify(text, level);
      setResult(data.simplified_text);
      toast('Text simplified!', 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Simplification failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Simplify Text</h1>
        <p>Adapt complex writing to a target reading level for broader accessibility.</p>
      </div>

      <div className="card" style={{ maxWidth: 760 }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(59,130,246,0.12)', color: 'var(--clr-info)' }}>
            <BookOpen size={20} />
          </div>
          <div>
            <div className="card-title">Text Simplifier</div>
            <div className="card-desc">Reduce reading complexity while keeping meaning intact</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <FormGroup label="Input Text">
            <textarea
              id="simplify-input"
              className="input"
              placeholder="Paste complex text to simplify…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
            />
          </FormGroup>
          <FormGroup label="Target Reading Level">
            <select
              id="simplify-level-select"
              className="input"
              value={level}
              onChange={(e) => setLevel(e.target.value)}
            >
              {LEVELS.map((l) => (
                <option key={l} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
              ))}
            </select>
          </FormGroup>
          <div style={{ display: 'flex', gap: 10 }}>
            <button id="simplify-submit-btn" className="btn btn-primary" type="submit" disabled={loading || !text.trim()}>
              {loading ? <Spinner /> : <ArrowRight size={16} />}
              {loading ? 'Simplifying…' : 'Simplify'}
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => { setText(''); setResult(''); }}>
              Clear
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4">
            <div className="form-label mb-2">Simplified Text</div>
            <ResultBox text={result} />
          </div>
        )}
      </div>
    </PageWrapper>
  );
}

