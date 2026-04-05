
/**
 * @file ChatArea.tsx
 * @description The main conversational interface that renders chat messages and input.
 * 
 * This component displays the conversation history with user and assistant messages,
 * source citations, and the input field for new messages. It follows the "Clean-First"
 * philosophy where assistant answers are rendered without highlights - verification
 * happens in the SourceViewer panel when users explicitly request it.
 * 
 * Key features:
 * - Auto-scrolling chat window
 * - Citation badges linking to source documents
 * - Collapsible citations when multiple sources present
 * - Enter to send, Shift+Enter for new line
 * - Loading states with visual feedback
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Send, FileText, BookOpen, ChevronDown, ChevronUp, Loader2, RotateCcw, HelpCircle, Search } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { Message, SourceChunk, ToolStep, Theme, HandbookDoc } from '../types';

/**
 * Props for ChatArea component
 */
interface ChatAreaProps {
  /** Array of messages in the current conversation */
  messages: Message[];
  /** Current value of the input text field */
  inputValue: string;
  /** Callback to update input value */
  setInputValue: (val: string) => void;
  /** Callback triggered when user sends a message */
  onSend: () => void;
  /** Callback triggered when user clicks retry on an error message */
  onRetry: () => void;
  /** Whether the AI is currently generating a response */
  isLoading: boolean;
  /** Callback to open source viewer for a specific document */
  onOpenSource: (sources: SourceChunk[], docId: string, messageId: string) => void;
  /** Current theme (light or dark) */
  theme: Theme;
  /** All loaded handbook documents (for citation lookups) */
  handbookDocs: HandbookDoc[];
  /** Callback to re-open the welcome/disclaimer modal */
  onOpenDisclaimer: () => void;
}

const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  inputValue,
  setInputValue,
  onSend,
  onRetry,
  isLoading,
  onOpenSource,
  theme,
  handbookDocs,
  onOpenDisclaimer
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isDark = theme === 'dark';
  const [expandedCitations, setExpandedCitations] = useState<Record<string, boolean>>({});
  const [faqOpen, setFaqOpen] = useState(false);

  const isTouchDevice = window.matchMedia('(pointer: coarse)').matches;
  const lineMetrics = useRef<{ oneLine: number; maxHeight: number }>({ oneLine: 0, maxHeight: 0 });

  // Cache line metrics once on mount
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    const computed = getComputedStyle(el);
    const lineHeight = parseFloat(computed.lineHeight) || 24;
    const paddingTop = parseFloat(computed.paddingTop) || 0;
    const paddingBottom = parseFloat(computed.paddingBottom) || 0;
    lineMetrics.current = {
      oneLine: lineHeight + paddingTop + paddingBottom,
      maxHeight: lineHeight * 3 + paddingTop + paddingBottom,
    };
  }, []);

  useEffect(() => {
    const el = textareaRef.current;
    const m = lineMetrics.current;
    if (!el || !m.oneLine) return;
    // Force shrink to 1 line so scrollHeight reflects true content
    el.style.height = m.oneLine + 'px';
    const desired = Math.min(el.scrollHeight, m.maxHeight);
    el.style.height = desired + 'px';
  }, [inputValue]);

  /**
   * Effect: Automatically scrolls the chat window to the bottom whenever
   * messages are updated or the loading state changes.
   */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  /**
   * Handles keyboard shortcuts in the textarea
   * - Enter: Send message
   * - Shift+Enter: New line
   */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isTouchDevice) {
      e.preventDefault();
      if (!isLoading) onSend();
    }
  };

  /**
   * Toggles citation expansion state for a specific message.
   * When collapsed, shows only the first source. When expanded, shows all sources.
   */
  const toggleCitations = (messageId: string) => {
    setExpandedCitations(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

  /**
   * Renders citation badges below assistant messages.
   * 
   * Each badge represents a unique handbook document that was cited in the response.
   * Clicking a badge opens the SourceViewer panel to that document with relevant
   * excerpts highlighted. Badges with highlights get a green indicator.
   * 
   * When multiple sources exist, only the first is shown by default with a
   * "Show more" button to expand the full list.
   * 
   * @param message - The assistant message with sources to render
   * @returns React element with citation badges, or null if no sources
   */
  const formatToolStep = (step: ToolStep): string => {
    if (step.tool_name === 'search_handbook') {
      const query = step.arguments?.query as string | undefined;
      return query ? `Searched handbook for "${query}"` : 'Searched handbook';
    }
    if (step.tool_name === 'send_feedback') return 'Sent feedback';
    if (step.tool_name === 'get_in_touch') return 'Sent contact request';
    return `Ran ${step.tool_name.replace(/_/g, ' ')}`;
  };

  const renderToolSteps = useCallback((message: Message) => {
    if (!message.toolSteps || message.toolSteps.length === 0) return null;

    return (
      <div className={`mb-3 text-xs rounded-lg border px-3 py-2 ${
        isDark
          ? 'border-zinc-800 bg-zinc-900/50 text-zinc-500'
          : 'border-zinc-100 bg-zinc-50 text-zinc-400'
      }`}>
        <div className="flex items-center gap-1.5 mb-1.5">
          <Search size={11} />
          <span className="font-medium">Steps</span>
        </div>
        <ol className="space-y-0.5 pl-4 list-decimal">
          {message.toolSteps
            .sort((a, b) => a.order - b.order)
            .map((step, i) => (
              <li key={i}>{formatToolStep(step)}</li>
            ))}
        </ol>
      </div>
    );
  }, [isDark]);

  /**
   * Renders a "Sources:" section below assistant messages.
   * Each source is numbered to match inline [n] citation markers.
   * Designed to support multiple source groups per message in the future.
   */
  const renderSources = useCallback((message: Message) => {
    if (!message.sources || message.sources.length === 0) return null;

    // Build sequentially numbered list of unique docs
    const entries: { num: number; docId: string; title: string }[] = [];
    const seenDocIds = new Set<string>();
    let num = 1;

    message.sources.forEach((s) => {
      if (!seenDocIds.has(s.docId)) {
        seenDocIds.add(s.docId);
        const doc = handbookDocs.find(d => d.id === s.docId);
        entries.push({ num: num++, docId: s.docId, title: doc?.title || s.docId.replace(/_/g, ' ') });
      }
    });

    const isExpanded = expandedCitations[message.id];
    const visibleEntries = isExpanded ? entries : entries.slice(0, 2);
    const hasMore = entries.length > 2;

    return (
      <div className={`mt-4 text-xs animate-in fade-in duration-700 ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
        <p className={`font-medium mb-1.5 ${isDark ? 'text-zinc-400' : 'text-zinc-500'}`}>Sources:</p>
        <div className="flex flex-col gap-1">
          {visibleEntries.map((entry) => (
            <button
              key={`${message.id}-src-${entry.num}`}
              onClick={() => onOpenSource(message.sources!, entry.docId, message.id)}
              className={`flex items-center gap-2 text-left transition-colors
                ${isDark ? 'hover:text-zinc-300' : 'hover:text-zinc-600'}`}
            >
              <span className={`shrink-0 w-4 h-4 flex items-center justify-center rounded-full text-[9px] font-medium
                ${isDark ? 'bg-zinc-800 text-zinc-500' : 'bg-zinc-200 text-zinc-500'}`}>
                {entry.num}
              </span>
              <FileText size={11} className="shrink-0" />
              <span className="truncate">{entry.title}</span>
            </button>
          ))}
        </div>
        {hasMore && (
          <button
            onClick={() => toggleCitations(message.id)}
            className={`mt-1.5 flex items-center gap-1 text-[11px] font-medium transition-colors
              ${isDark ? 'text-zinc-600 hover:text-zinc-400' : 'text-zinc-400 hover:text-zinc-600'}`}
          >
            {isExpanded ? (
              <>Show less <ChevronUp size={11} /></>
            ) : (
              <>+{entries.length - 2} more <ChevronDown size={11} /></>
            )}
          </button>
        )}
      </div>
    );
  }, [expandedCitations, isDark, onOpenSource, handbookDocs]);

  return (
    <div className={`flex-1 flex flex-col relative h-full min-w-0 transition-all duration-300
      ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-8 sm:py-16 w-full scroll-smooth"
      >
        <div className="max-w-3xl mx-auto w-full">
          {/* Welcome Screen: Displays common questions if no messages exist */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center text-center space-y-8 animate-in fade-in duration-1000 py-12">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shadow-sm border transition-colors shrink-0
                  ${isDark ? 'bg-zinc-800 border-zinc-700 text-emerald-500' : 'bg-emerald-500 border-emerald-400 text-white'}`}>
                  <BookOpen size={20} />
                </div>
                <h1 className={`text-2xl font-semibold tracking-tight ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}>
                  Welcome to Nexus
                </h1>
              </div>

              <div className="space-y-2">
                <p className={`${isDark ? 'text-zinc-400' : 'text-zinc-500'} text-sm max-w-sm mx-auto`}>
                  Ask anything about company policies, benefits, or workspace guidelines.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
                {[
                  "Tell me more about this app",
                  "What cycling benefits do we have?",
                  "Tell me about parental leave",
                  "What a Lead Data Engineer does?"
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setInputValue(q);
                    }}
                    className={`p-4 text-left text-sm rounded-xl border transition-all
                      ${isDark
                        ? 'bg-zinc-900/50 border-zinc-800 hover:border-zinc-700 text-zinc-400'
                        : 'bg-white border-zinc-200 hover:border-emerald-500/30 hover:bg-emerald-50/10 text-zinc-600'}`}
                  >
                    {q}
                  </button>
                ))}
              </div>

              {/* FAQ Section */}
              <div className="w-full max-w-xl">
                <button
                  onClick={() => setFaqOpen(!faqOpen)}
                  className={`flex items-center gap-2 mx-auto text-xs font-medium transition-colors
                    ${isDark ? 'text-zinc-500 hover:text-zinc-300' : 'text-zinc-400 hover:text-zinc-600'}`}
                >
                  <HelpCircle size={14} />
                  What can I help with?
                  {faqOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>

                {faqOpen && (
                  <div className={`mt-4 text-left text-xs space-y-3 rounded-xl border p-4 animate-in fade-in duration-300
                    ${isDark ? 'bg-zinc-900/50 border-zinc-800 text-zinc-400' : 'bg-zinc-50 border-zinc-200 text-zinc-500'}`}>
                    {[
                      { title: 'Benefits & Compensation', examples: ['Pension scheme & employer matching', 'Holiday allowance & booking', 'Cycle to work, tech loans, medical insurance', 'Learning budgets & development'] },
                      { title: 'Roles & Careers', examples: ['50+ role descriptions across all levels', 'SFIA framework & career progression', 'Responsibilities, competencies & expectations'] },
                      { title: 'Ways of Working', examples: ['Delivery standards & development practices', 'Onboarding, 1-on-1s, probation, promotions', 'Hybrid working & office policies'] },
                      { title: 'Policies & Security', examples: ['Password, BYOD & data protection policies', 'Anti-corruption, whistleblowing, EDI', 'Laptop specs, VPN, security clearance'] },
                      { title: 'Welfare & Leave', examples: ['Sick leave, parental leave, paid counselling', 'Mental health support & raising issues'] },
                    ].map((cat) => (
                      <div key={cat.title}>
                        <p className={`font-semibold text-xs mb-1 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>{cat.title}</p>
                        <ul className="space-y-0.5 pl-3">
                          {cat.examples.map((ex) => (
                            <li key={ex} className="list-disc">{ex}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                    <p className={`pt-2 border-t text-[10px] ${isDark ? 'border-zinc-800 text-zinc-600' : 'border-zinc-200 text-zinc-400'}`}>
                      Based on the Made Tech handbook (Jan 2026 snapshot). Cannot answer about specific salaries, client details, or org structure.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="space-y-12">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex flex-col animate-in fade-in slide-in-from-bottom-2 duration-500 ${message.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed
                  ${message.role === 'user'
                    ? `max-w-[85%] sm:max-w-[80%] ${isDark ? 'bg-zinc-800 text-zinc-100' : 'bg-zinc-100 text-zinc-900'}`
                    : `w-full min-w-0 ${isDark ? 'text-zinc-300' : 'text-zinc-800'}`}`}
                >
                  {message.role === 'assistant' ? (
                    <>
                    {renderToolSteps(message)}
                    <MarkdownRenderer
                      content={message.content}
                      theme={theme}
                      sources={message.sources}
                      onCiteClick={message.sources ? (docId) => onOpenSource(message.sources!, docId, message.id) : undefined}
                    />
                    </>
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
                {message.isError && (
                  <button
                    onClick={onRetry}
                    className={`mt-2 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border
                      ${isDark
                        ? 'border-zinc-700 text-zinc-400 hover:border-emerald-500/50 hover:text-emerald-400'
                        : 'border-zinc-200 text-zinc-500 hover:border-emerald-500 hover:text-emerald-600'}`}
                  >
                    <RotateCcw size={12} />
                    Try again
                  </button>
                )}
                {message.role === 'assistant' && !message.isError && renderSources(message)}
              </div>
            ))}

            {isLoading && (
              <div className="flex items-center gap-3 animate-pulse">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isDark ? 'bg-zinc-800' : 'bg-zinc-100'}`}>
                  <Loader2 size={16} className="animate-spin text-emerald-500" />
                </div>
                <div className="text-xs font-medium text-emerald-500 tracking-wide uppercase">AI is analyzing handbook...</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className={`p-4 sm:p-6 border-t ${isDark ? 'border-zinc-800 bg-zinc-950/50' : 'border-zinc-100 bg-white/50'} backdrop-blur-md`}>
        <div className={`max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border px-4 py-2 transition-all
          ${isDark
            ? 'border-zinc-800 bg-zinc-950 focus-within:border-zinc-700'
            : 'border-zinc-200 bg-white shadow-sm focus-within:border-emerald-500 focus-within:shadow-emerald-500/10'}`}>
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search handbook documents..."
            enterKeyHint={isTouchDevice ? 'enter' : 'send'}
            rows={1}
            className={`flex-1 bg-transparent py-2 text-base sm:text-sm outline-none resize-none overflow-y-auto overscroll-contain leading-6
              ${isDark ? 'text-zinc-200 placeholder:text-zinc-600' : 'text-zinc-900 placeholder:text-zinc-400'}`}
          />
          <button
            onClick={onSend}
            disabled={!inputValue.trim() || isLoading}
            className={`shrink-0 mb-0.5 p-2 rounded-xl transition-all
              ${!inputValue.trim() || isLoading
                ? `cursor-not-allowed ${isDark ? 'text-zinc-600' : 'text-zinc-300'}`
                : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-lg shadow-emerald-500/20 active:scale-95'}`}
          >
            {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
        <p className={`mt-2 text-center text-[10px] ${isDark ? 'text-zinc-600' : 'text-zinc-400'}`}>
          AI results may be subject to human error. Please verify critical policy details in original documents.{' '}
          <button
            onClick={onOpenDisclaimer}
            className={`underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-500 hover:text-emerald-400' : 'text-emerald-600 hover:text-emerald-500'}`}
          >
            Disclaimer
          </button>
        </p>
      </div>
    </div>
  );
};

// Add default export to resolve "no default export" error in App.tsx
export default React.memo(ChatArea);
