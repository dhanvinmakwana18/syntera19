import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ChatView from './pages/ChatView';
import KnowledgeBaseView from './pages/KnowledgeBaseView';
import { Terminal, Database, Activity, Settings } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-[#09090b] text-zinc-300 font-sans selection:bg-zinc-800">
        {/* Sidebar Navigation */}
        <aside className="w-16 sm:w-64 border-r border-zinc-800/60 bg-zinc-950/50 flex flex-col items-center sm:items-start py-6 px-0 sm:px-4 backdrop-blur-xl transition-all">
          <div className="mb-10 sm:px-2 flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <Terminal size={18} />
            </div>
            <h1 className="hidden sm:block font-bold text-white tracking-tight">Syntera</h1>
          </div>
          
          <nav className="flex-1 w-full space-y-2">
            <NavItem to="/" icon={<Terminal size={20} />} label="Laboratory" />
            <NavItem to="/knowledge" icon={<Database size={20} />} label="Knowledge Base" />
            <NavItem to="/observability" icon={<Activity size={20} />} label="Observability" />
          </nav>
          
          <div className="w-full mt-auto">
             <NavItem to="/settings" icon={<Settings size={20} />} label="Settings" />
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 h-full overflow-hidden relative">
          <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/10 rounded-full blur-[100px] pointer-events-none -z-10" />
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none -z-10" />
          
          <Routes>
            <Route path="/" element={<ChatView />} />
            <Route path="/knowledge" element={<KnowledgeBaseView />} />
            {/* Placeholders for other routes */}
            <Route path="/observability" element={<div className="p-8">Observability Dashboard (Coming Soon)</div>} />
            <Route path="/settings" element={<div className="p-8">System Settings (Coming Soon)</div>} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

function NavItem({ to, icon, label }: { to: string, icon: React.ReactNode, label: string }) {
  return (
    <Link to={to} className="flex items-center gap-3 px-0 sm:px-3 py-3 w-full rounded-xl hover:bg-zinc-800/50 text-zinc-400 hover:text-white transition-colors justify-center sm:justify-start">
      {icon}
      <span className="hidden sm:inline text-sm font-medium">{label}</span>
    </Link>
  );
}

export default App;
