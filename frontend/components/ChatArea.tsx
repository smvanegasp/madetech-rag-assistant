
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
import { Send, FileText, BookOpen, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { Message, SourceChunk, Theme, HandbookDoc } from '../types';

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

  const MAX_TEXTAREA_HEIGHT = 128;

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT) + 'px';
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
    if (e.key === 'Enter' && !e.shiftKey) {
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
  const renderCitations = useCallback((message: Message) => {
    if (!message.sources || message.sources.length === 0) return null;

    const distinctDocs: string[] = [];
    const seenDocIds = new Set<string>();
    
    message.sources.forEach(s => {
      if (!seenDocIds.has(s.docId)) {
        distinctDocs.push(s.docId);
        seenDocIds.add(s.docId);
      }
    });

    const cappedDocs = distinctDocs.slice(0, 4);
    const isExpanded = expandedCitations[message.id];
    const visibleDocIds = isExpanded ? cappedDocs : [cappedDocs[0]];
    const hasMore = cappedDocs.length > 1;

    return (
      <div className="flex flex-col mt-6 animate-in fade-in duration-700">
        <div className="flex flex-wrap gap-2 items-center">
          {visibleDocIds.map((docId, idx) => {
            const doc = handbookDocs.find(d => d.id === docId);
            return doc ? (
              <button
                key={`${message.id}-${idx}`}
                onClick={() => onOpenSource(message.sources!, docId, message.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-all border
                  ${isDark 
                    ? 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-emerald-500/50 hover:text-zinc-200' 
                    : 'bg-white border-zinc-200 text-zinc-600 hover:border-emerald-500 hover:text-emerald-600'}`}
              >
                <FileText size={12} className="text-zinc-400" />
                <span className="truncate max-w-[150px] font-medium">{doc.title}</span>
              </button>
            ) : null;
          })}

          {hasMore && (
            <button
              onClick={() => toggleCitations(message.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border
                ${isDark 
                  ? 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200' 
                  : 'bg-zinc-100 border-zinc-200 text-zinc-600 hover:bg-zinc-200'}`}
            >
              {isExpanded ? (
                <>Show less <ChevronUp size={12} /></>
              ) : (
                <>+{cappedDocs.length - 1} more sources <ChevronDown size={12} /></>
              )}
            </button>
          )}
        </div>
      </div>
    );
  }, [expandedCitations, isDark, onOpenSource, handbookDocs]);

  return (
    <div className={`flex-1 flex flex-col relative h-full transition-all duration-300
      ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-8 sm:py-16 w-full scroll-smooth"
      >
        <div className="max-w-3xl mx-auto w-full">
          {/* Welcome Screen: Displays common questions if no messages exist */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-8 animate-in fade-in duration-1000 py-12">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-sm border transition-colors
                ${isDark ? 'bg-zinc-800 border-zinc-700 text-emerald-500' : 'bg-emerald-500 border-emerald-400 text-white'}`}>
                <BookOpen size={24} />
              </div>
              
              <div className="space-y-2">
                <h1 className={`text-2xl font-semibold tracking-tight ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}>
                  Knowledge Search
                </h1>
                <p className={`${isDark ? 'text-zinc-400' : 'text-zinc-500'} text-sm max-w-sm mx-auto`}>
                  Ask anything about company policies, benefits, or workspace guidelines.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
                {[
                  "Which cities does Made Tech have offices in?",
                  "How do I get confidential advice if I'm struggling?",
                  "Do all employees sign confidentiality agreements?",
                  "How often are laptops replaced for non-engineer roles?"
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
            </div>
          )}

          {/* Messages */}
          <div className="space-y-12">
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={`flex flex-col animate-in fade-in slide-in-from-bottom-2 duration-500 ${message.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div className={`max-w-[85%] sm:max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed
                  ${message.role === 'user' 
                    ? (isDark ? 'bg-zinc-800 text-zinc-100' : 'bg-zinc-100 text-zinc-900') 
                    : (isDark ? 'text-zinc-300' : 'text-zinc-800')}`}
                >
                  {message.role === 'assistant' ? (
                    <MarkdownRenderer content={message.content} theme={theme} />
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
                {message.role === 'assistant' && renderCitations(message)}
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
        <div className={`max-w-3xl mx-auto flex flex-col rounded-2xl border overflow-hidden transition-all
          ${isDark 
            ? 'border-zinc-800 bg-zinc-950 focus-within:border-zinc-700' 
            : 'border-zinc-200 bg-white shadow-sm focus-within:border-emerald-500 focus-within:shadow-emerald-500/10'}`}>
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search handbook documents..."
            rows={1}
            className={`w-full bg-transparent pl-4 pr-4 pt-3.5 pb-1 text-sm outline-none resize-none overflow-y-auto
              ${isDark ? 'text-zinc-200 placeholder:text-zinc-600' : 'text-zinc-900 placeholder:text-zinc-400'}`}
            style={{ maxHeight: `${MAX_TEXTAREA_HEIGHT}px` }}
          />
          <div className="flex justify-end px-2 pb-2">
            <button
              onClick={onSend}
              disabled={!inputValue.trim() || isLoading}
              className={`p-2 rounded-xl transition-all
                ${!inputValue.trim() || isLoading
                  ? 'opacity-30 cursor-not-allowed'
                  : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-lg shadow-emerald-500/20 active:scale-95'}`}
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
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
