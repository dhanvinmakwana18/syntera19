import React, { useState, useRef, useEffect } from 'react';
import { Send, Cpu, Layout, FileText, ChevronRight, ShieldCheck, ShieldAlert, Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { chatWithSyntera, getHealth } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';

type Message = {
  role: string;
  content: string;
  sources?: any[];
  trace?: any[];
  grounded?: boolean;
  routing_mode?: string;
  error_code?: string;
};

export default function ChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('auto');
  const [health, setHealth] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Health check on mount and every 30 seconds
  useEffect(() => {
    const checkHealth = async () => {
      const h = await getHealth();
      setHealth(h);
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuery = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setLoading(true);

    try {
      const response = await chatWithSyntera(userQuery, mode);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        trace: response.trace,
        grounded: response.grounded,
        routing_mode: response.routing_mode,
      }]);
    } catch (error: any) {
      const errData = error?.response?.data;
      const errorCode = errData?.error_code || 'BACKEND_OFFLINE';
      const errorMsg = errData?.detail || 'Failed to connect to the Syntera backend engine.';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ ${errorMsg}`,
        error_code: errorCode,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const healthStatus = health?.status || 'OFFLINE';
  const healthColor = healthStatus === 'ONLINE' ? 'text-emerald-400' : healthStatus === 'DEGRADED' ? 'text-amber-400' : 'text-rose-400';
  const HealthIcon = healthStatus === 'ONLINE' ? Wifi : healthStatus === 'DEGRADED' ? AlertCircle : WifiOff;

  return (
    <div className="h-full flex">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col relative h-full">
        {/* Header */}
        <div className="h-16 border-b border-zinc-800/60 bg-zinc-950/30 backdrop-blur-md flex items-center px-6 justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-white">Syntera Engine</h2>
            <div className={`flex items-center gap-1.5 text-[10px] font-mono ${healthColor} px-2 py-1 rounded-full border border-current/20 bg-current/5`}>
              <HealthIcon size={10} />
              {healthStatus}
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-zinc-500">ROUTING MODE:</span>
            <select 
              value={mode} 
              onChange={(e) => setMode(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-zinc-300 focus:outline-none focus:border-blue-500"
            >
              <option value="auto">AUTO</option>
              <option value="direct">DIRECT</option>
              <option value="rag">RAG</option>
              <option value="agentic">AGENTIC</option>
            </select>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center text-center px-4">
              <div className="max-w-md">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-500/20 to-purple-500/20 border border-zinc-800 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-blue-900/10">
                  <Cpu className="text-blue-400" size={32} />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Autonomous Agentic Reasoning</h3>
                <p className="text-zinc-400 text-sm">
                  Initialize query to test document retrieval, tool calling, and deterministic routing.
                </p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl p-4 ${
                msg.role === 'user' 
                  ? 'bg-blue-600/10 border border-blue-500/20 text-blue-100' 
                  : msg.error_code
                    ? 'bg-rose-950/30 border border-rose-800/40 text-rose-200'
                    : 'bg-zinc-900/80 border border-zinc-800/80 text-zinc-300'
              }`}>
                {/* Routing mode badge */}
                {msg.routing_mode && (
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      {msg.routing_mode}
                    </span>
                    {msg.grounded !== undefined && (
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded flex items-center gap-1 ${
                        msg.grounded 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      }`}>
                        {msg.grounded ? <ShieldCheck size={10} /> : <ShieldAlert size={10} />}
                        {msg.grounded ? 'GROUNDED' : 'UNGROUNDED'}
                      </span>
                    )}
                  </div>
                )}
                
                {/* Error code badge */}
                {msg.error_code && (
                  <div className="mb-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      {msg.error_code}
                    </span>
                  </div>
                )}
                
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</div>
                
                {/* Sources Display */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-zinc-800">
                    <div className="text-xs font-mono text-zinc-500 mb-2 flex items-center gap-1">
                      <FileText size={12} /> GROUNDING SOURCES ({msg.sources.length})
                    </div>
                    <div className="space-y-2">
                      {msg.sources.map((src, idx) => (
                        <div key={idx} className="bg-zinc-950/50 p-2 rounded-lg border border-zinc-800/50 text-xs">
                          <span className="text-blue-400 font-semibold">[{src.id}] {src.filename}</span>
                          {src.page !== '?' && <span className="text-zinc-600 ml-2">p.{src.page}</span>}
                          <span className="text-zinc-600 ml-2">Score: {src.score?.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-zinc-900/80 border border-zinc-800/80 rounded-2xl p-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse delay-75"></span>
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse delay-150"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <div className="p-4 border-t border-zinc-800/60 bg-zinc-950/50 backdrop-blur-md">
          <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Query the Syntera Engine..."
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl pl-4 pr-12 py-4 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all text-white placeholder-zinc-500 shadow-inner"
              disabled={loading}
            />
            <button 
              type="submit" 
              disabled={!input.trim() || loading}
              className="absolute right-2 top-2 p-2 bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 hover:text-blue-300 rounded-lg transition-colors disabled:opacity-50"
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>

      {/* Observability Sidebar (Right) */}
      <div className="w-80 border-l border-zinc-800/60 bg-zinc-950/50 backdrop-blur-xl hidden lg:flex flex-col">
        <div className="h-16 border-b border-zinc-800/60 flex items-center px-4 gap-2 text-sm font-semibold text-white">
          <Layout size={16} className="text-purple-400" /> Execution Trace
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <AnimatePresence>
            {messages.filter(m => m.trace).map((msg, idx) => (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="mb-6"
              >
                <div className="text-[10px] font-mono text-zinc-500 mb-3 uppercase">Transaction #{idx+1}</div>
                <div className="space-y-3 relative before:absolute before:inset-0 before:ml-[11px] before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-zinc-800 before:to-transparent">
                  {msg.trace?.map((step: any, sIdx: number) => (
                    <div key={sIdx} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                      <div className={`flex items-center justify-center w-6 h-6 rounded-full border shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10 ${
                        step.step?.includes('ERROR') 
                          ? 'border-rose-700 bg-rose-900 text-rose-400'
                          : step.step?.includes('GROUNDING')
                            ? 'border-emerald-700 bg-emerald-900 text-emerald-400'
                            : 'border-zinc-700 bg-zinc-900 text-zinc-400'
                      }`}>
                        <ChevronRight size={12} />
                      </div>
                      <div className="w-[calc(100%-2rem)] md:w-[calc(50%-1.5rem)] p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/50 shadow-sm text-xs">
                        <div className={`font-mono mb-1 ${
                          step.step?.includes('ERROR') ? 'text-rose-400' : 'text-blue-400'
                        }`}>{step.step}</div>
                        <div className="text-zinc-400">{step.action}</div>
                        {step.latency_ms !== undefined && (
                          <div className="text-zinc-600 font-mono mt-1 text-[10px]">{step.latency_ms}ms</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center text-zinc-600 text-xs font-mono text-center">
              Awaiting query execution...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
