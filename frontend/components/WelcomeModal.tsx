import React, { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { Theme } from '../types';

interface WelcomeModalProps {
  isOpen: boolean;
  /** True when it opened by itself (first visit): it then closes by itself after AUTO_CLOSE_MS. */
  auto: boolean;
  onClose: () => void;
  theme: Theme;
}

export const AUTO_CLOSE_MS = 20_000;
const TICK_MS = 100;

/**
 * What Nexus is, as a popup (same behaviour as the Car Haul app's intro). On the first visit it
 * closes by itself after 20 s; hovering it or using the keyboard in it pauses the countdown, so
 * nobody loses it mid-read. Reopened from the header button, it stays until closed.
 */
const WelcomeModal: React.FC<WelcomeModalProps> = ({ isOpen, auto, onClose, theme }) => {
  const isDark = theme === 'dark';
  const [left, setLeft] = useState(AUTO_CLOSE_MS);
  const [paused, setPaused] = useState(false);
  const button = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isOpen) {
      setLeft(AUTO_CLOSE_MS);
      setPaused(false);
      button.current?.focus({ preventScroll: true });
    }
  }, [isOpen]);

  // Counts down in small steps; the bar's CSS transition fills the gaps, so it moves smoothly.
  useEffect(() => {
    if (!isOpen || !auto || paused) return;
    const id = setInterval(() => setLeft(l => l - TICK_MS), TICK_MS);
    return () => clearInterval(id);
  }, [isOpen, auto, paused]);

  useEffect(() => {
    if (isOpen && auto && left <= 0) onClose();
  }, [isOpen, auto, left, onClose]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const link = `underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-500'}`;

  return (
    // Bottom sheet on phones, centred card from sm up.
    <div
      className="fixed inset-0 z-[100] flex items-end sm:items-center justify-center p-4 animate-in fade-in duration-300"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" aria-hidden="true" />

      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="welcome-title"
        onClick={e => e.stopPropagation()}
        onPointerEnter={() => setPaused(true)}
        onPointerLeave={() => setPaused(false)}
        onKeyDown={e => e.key !== 'Escape' && setPaused(true)}
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
            onClick={onClose}
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
            <a href="https://github.com/madetech/handbook" target="_blank" rel="noopener noreferrer" className={link}>
              Made Tech Handbook
            </a>
            {' '}(Jan 2026 snapshot).
          </p>

          <div className={`flex flex-wrap items-center gap-x-3 gap-y-1 text-xs ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
            <span>Built by{' '}
              <a href="https://www.linkedin.com/in/sergio-vanegas/" target="_blank" rel="noopener noreferrer" className={link}>Sergio Vanegas</a>
            </span>
            <span>·</span>
            <a href="https://github.com/smvanegasp/madetech-rag-assistant" target="_blank" rel="noopener noreferrer" className={link}>View source code</a>
            <span>·</span>
            <a href="https://sergiovanegas.com/projects/" className={link}>More projects</a>
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

          {/* Countdown + confirm */}
          <div className="flex items-center justify-between gap-3">
            <span className={`text-xs ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`} aria-live="off">
              {auto ? (paused ? 'Paused' : `Closes in ${Math.max(0, Math.ceil(left / 1000))} s`) : ''}
            </span>
            <button
              ref={button}
              type="button"
              onClick={onClose}
              className="rounded-lg bg-emerald-500 hover:bg-emerald-600 active:scale-95 px-5 py-2 text-sm font-semibold text-white transition
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2"
            >
              Got it
            </button>
          </div>
        </div>

        {/* Time left: shrinks in 100 ms steps, smoothed by the width transition. */}
        {auto && (
          <div className={`h-1 w-full ${isDark ? 'bg-zinc-800' : 'bg-zinc-100'}`} aria-hidden="true">
            <div
              className="h-full bg-emerald-500 transition-[width] duration-100 ease-linear"
              style={{ width: `${Math.max(0, left / AUTO_CLOSE_MS) * 100}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default WelcomeModal;
