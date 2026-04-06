import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { Theme } from '../types';

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: Theme;
}

const AUTO_DISMISS_MS = 45_000;

const WelcomeModal: React.FC<WelcomeModalProps> = ({ isOpen, onClose, theme }) => {
  const isDark = theme === 'dark';

  const dismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimers = () => {
    if (dismissTimer.current) clearTimeout(dismissTimer.current);
  };

  useEffect(() => {
    if (!isOpen) return;

    dismissTimer.current = setTimeout(() => {
      onClose();
    }, AUTO_DISMISS_MS);

    return clearTimers;
  }, [isOpen, onClose]);

  const handleClose = () => {
    clearTimers();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300"
      aria-modal="true"
      role="dialog"
      aria-labelledby="welcome-title"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Card */}
      <div
        className={`relative z-10 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden
          ${isDark ? 'bg-zinc-900 text-zinc-200' : 'bg-white text-zinc-900'}
          animate-in zoom-in-95 fade-in duration-300`}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 pt-6 pb-4">
          <div>
            <h2
              id="welcome-title"
              className={`text-base font-semibold tracking-tight ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}
            >
              Welcome to Nexus
            </h2>
            <p className={`mt-0.5 text-xs ${isDark ? 'text-emerald-400' : 'text-emerald-600'}`}>
              Academic project · Non-commercial
            </p>
          </div>
          <button
            onClick={handleClose}
            aria-label="Close"
            className={`ml-4 mt-0.5 shrink-0 p-1.5 rounded-lg transition-colors
              ${isDark ? 'hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300' : 'hover:bg-zinc-100 text-zinc-400 hover:text-zinc-600'}`}
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className={`px-6 pb-5 space-y-4 text-sm leading-relaxed ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
          <p>
            This assistant helps Made Tech employees explore company policies, benefits, roles, and ways of working by searching the{' '}
            <a
              href="https://github.com/madetech/handbook"
              target="_blank"
              rel="noopener noreferrer"
              className={`underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-500'}`}
            >
              Made Tech Handbook
            </a>
            {' '}(Jan 2026 snapshot).
          </p>

          <div className={`flex items-center gap-3 text-xs ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
            <span>Built by{' '}
              <a
                href="https://www.linkedin.com/in/sergio-vanegas/"
                target="_blank"
                rel="noopener noreferrer"
                className={`underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-500'}`}
              >Sergio Vanegas</a>
            </span>
            <span>·</span>
            <a
              href="https://github.com/smvanegasp/madetech-rag-assistant"
              target="_blank"
              rel="noopener noreferrer"
              className={`underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-500'}`}
            >View source code</a>
          </div>

          <div className={`rounded-xl p-4 text-xs space-y-2.5 ${isDark ? 'bg-zinc-800/60 text-zinc-400' : 'bg-zinc-50 text-zinc-500'}`}>
            <p className={`font-medium ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>Why this exists</p>
            <p>Nobody reads HR policies until they need a quick answer. The usual path — searching folders, emailing HR, asking colleagues — is slow and often outdated. Nexus is an <strong className={isDark ? 'text-zinc-300' : 'text-zinc-600'}>agentic RAG chatbot</strong> that searches 161 handbook documents using semantic and keyword search.</p>
            <ul className="space-y-0.5 list-none">
              {[
                ['24/7 availability', 'Answers when you need them, not when HR is online.'],
                ['Grounded answers', 'Every response referenced in official documents.'],
                ['Conversational', 'Ask follow-ups naturally, like chatting with a colleague.'],
              ].map(([title, desc], i) => (
                <li key={i} className="flex gap-2">
                  <span className="mt-0.5 shrink-0 text-emerald-500">·</span>
                  <span><strong className={isDark ? 'text-zinc-300' : 'text-zinc-600'}>{title}.</strong> {desc}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Auto-dismiss progress bar */}
        <style>{`
          @keyframes welcome-progress {
            from { width: 0% }
            to   { width: 100% }
          }
        `}</style>
        <div className={`h-0.5 w-full ${isDark ? 'bg-zinc-800' : 'bg-zinc-100'}`}>
          <div
            className="h-full bg-emerald-500"
            style={{
              animation: `welcome-progress ${AUTO_DISMISS_MS}ms linear forwards`,
            }}
            aria-hidden="true"
          />
        </div>
      </div>
    </div>
  );
};

export default WelcomeModal;
