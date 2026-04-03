import { useState } from 'react';
import './index.css';
import { ToastProvider } from './context/ToastContext';
import Topbar from './components/Topbar';
import Sidebar from './components/Sidebar';
import DashboardPage  from './pages/DashboardPage';
import ParaphrasePage from './pages/ParaphrasePage';
import GrammarPage    from './pages/GrammarPage';
import SimplifyPage   from './pages/SimplifyPage';
import TonePage       from './pages/TonePage';
import SummarizePage  from './pages/SummarizePage';
import RagPage        from './pages/RagPage';
import { AnimatePresence } from 'framer-motion';

// page → component map
const PAGES = {
  dashboard:  DashboardPage,
  paraphrase: ParaphrasePage,
  grammar:    GrammarPage,
  simplify:   SimplifyPage,
  tone:       TonePage,
  summarize:  SummarizePage,
  rag:        RagPage,
};

export default function App() {
  const [active, setActive] = useState('dashboard');
  const Page = PAGES[active] ?? DashboardPage;

  return (
    <ToastProvider>
      <div className="bg-blobs" />
      <div className="app-shell">
        <Topbar />
        <Sidebar active={active} setActive={setActive} />
        <main className="main-content">
          <AnimatePresence mode="wait">
            <Page key={active} setActive={setActive} />
          </AnimatePresence>
        </main>
      </div>
    </ToastProvider>
  );
}
