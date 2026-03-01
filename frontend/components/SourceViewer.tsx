/**
 * @file SourceViewer.tsx
 * @description Sliding side panel for inspecting document sources. 
 * Features a dual-view system: 
 * 1. "Excerpts" (Chunk Mode): Displays the raw verbatim snippets cited in the AI's response.
 * 2. "Full Doc" (Full Mode): Displays the entire handbook document.
 */

import React, { useMemo } from 'react';
import { X, FileText, ChevronLeft, ChevronRight, Info, Loader2, Sparkles } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { SelectedSource, ViewMode, Theme, HandbookDoc, Chat } from '../types';

interface SourceViewerProps {
  isOpen: boolean;
  onClose: () => void;
  source: SelectedSource | null;
  onViewModeChange: (mode: ViewMode) => void;
  onDocChange: (docId: string) => void;
  theme: Theme;
  activeChat: Chat | undefined;
  /** Global loading state for this specific document in the current context */
  isDocAnalyzing: boolean;
  /** Callback to trigger background analysis for the current source document */
  onTriggerAnalysis: () => void;
  handbookDocs: HandbookDoc[];
}

const SourceViewer: React.FC<SourceViewerProps> = ({
  isOpen,
  onClose,
  source,
  onViewModeChange,
  onDocChange,
  theme,
  activeChat,
  isDocAnalyzing,
  onTriggerAnalysis,
  handbookDocs
}) => {
  // Destructure state with fallback defaults (defaulting to 'full' view)
  const { sources = [], currentDocId = '', viewMode = 'full', contextMessageId = '' } = source || {};

  // Find the specific handbook document from backend data
  const doc = handbookDocs.find(d => d.id === currentDocId) as HandbookDoc | undefined;
  const isDark = theme === 'dark';

  // Retrieve existing AI analysis for this specific message-document pair
  const contextMessage = activeChat?.messages.find(m => m.id === contextMessageId);
  const relevanceHighlights = contextMessage?.highlights?.[currentDocId];

  // Navigation helpers for multi-document citations
  const currentDocCitations = sources.filter(s => s.docId === currentDocId);
  const distinctDocIds = Array.from(new Set(sources.map(s => s.docId)));
  const currentDocIndex = distinctDocIds.indexOf(currentDocId);

  /**
   * FIX logic: We only provide highlight targets to the MarkdownRenderer IF
   * the user has "pushed for highlights" (i.e., relevanceHighlights exists).
   * Otherwise, the document remains clean.
   */
  const allHighlightTargets = useMemo(() => {
    // If the user hasn't triggered AI analysis, return empty to keep document clean
    if (!relevanceHighlights) return [];

    // Once analyzed, combine both original RAG snippets and new semantic context
    const explicitSnippets = currentDocCitations.map(c => c.snippet);
    return Array.from(new Set([...explicitSnippets, ...relevanceHighlights]))
      .filter(t => t && t.trim().length > 3);
  }, [currentDocCitations, relevanceHighlights]);

  if (!isOpen) return null;

  return (
    <div
      className={`fixed top-0 right-0 h-full w-full sm:w-[500px] lg:w-[600px] z-[60] shadow-2xl transition-transform duration-500 ease-in-out border-l
        ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        ${isDark ? 'bg-zinc-950 border-zinc-800 text-zinc-200' : 'bg-zinc-50 border-zinc-200 text-zinc-900'}`}
      role="complementary"
      aria-label="Document Viewer Panel"
    >
      <div className="flex flex-col h-full">
        {/* Panel Header */}
        <div className={`p-4 flex items-center justify-between border-b ${isDark ? 'border-zinc-800 bg-zinc-900/50' : 'border-zinc-200 bg-white'}`}>
          <div className="flex items-center gap-3 min-w-0">
            <div className={`p-2 rounded-lg ${isDark ? 'bg-zinc-800 text-emerald-500' : 'bg-emerald-50 text-emerald-600'}`}>
              <FileText size={18} />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm font-bold truncate">{doc?.title || 'Document View'}</h2>
              <p className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold">{doc?.category}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close viewer"
            className={`p-2 rounded-lg transition-colors ${isDark ? 'hover:bg-zinc-800 text-zinc-500' : 'hover:bg-zinc-100 text-zinc-500'}`}
          >
            <X size={18} />
          </button>
        </div>

        {/* Toolbar: Document Navigation and AI Trigger */}
        <div className={`px-4 py-2 flex items-center justify-between border-b text-xs ${isDark ? 'border-zinc-800 bg-zinc-900/30' : 'border-zinc-100 bg-zinc-50/50'}`}>
          <div className="flex items-center gap-1">
            <button
              disabled={currentDocIndex <= 0}
              onClick={() => onDocChange(distinctDocIds[currentDocIndex - 1])}
              className="p-1.5 disabled:opacity-30 hover:bg-zinc-500/10 rounded transition-colors"
              aria-label="Previous document in citations"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="font-medium px-1 text-zinc-500 tabular-nums">
              {currentDocIndex + 1} of {distinctDocIds.length}
            </span>
            <button
              disabled={currentDocIndex >= distinctDocIds.length - 1}
              onClick={() => onDocChange(distinctDocIds[currentDocIndex + 1])}
              className="p-1.5 disabled:opacity-30 hover:bg-zinc-500/10 rounded transition-colors"
              aria-label="Next document in citations"
            >
              <ChevronRight size={14} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            {/* AI Highlight Section: Integrated into toolbar */}
            <div className="flex items-center">
              {isDocAnalyzing ? (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider bg-zinc-500/10 text-emerald-500">
                  <Loader2 size={12} className="animate-spin" />
                  Analyzing
                </div>
              ) : (
                <>
                  {!relevanceHighlights ? (
                    <button
                      onClick={onTriggerAnalysis}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all
                        ${isDark ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300' : 'bg-zinc-100 hover:bg-zinc-200 text-zinc-600'}`}
                    >
                      <Sparkles size={12} className="text-emerald-500" />
                      Highlight with AI
                    </button>
                  ) : (
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald-500/10 text-emerald-500 text-[9px] font-bold uppercase tracking-widest animate-in fade-in zoom-in-95 duration-300">
                      <Sparkles size={10} />
                      <span>Highlights Active</span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Mode Toggle: DEACTIVATED - keeping code for potential future use */}
            {/* <div className="flex bg-zinc-500/10 p-1 rounded-lg">
              <button
                onClick={() => onViewModeChange('chunk')}
                className={`px-3 py-1 rounded-md transition-all ${viewMode === 'chunk' ? (isDark ? 'bg-zinc-800 text-white shadow-sm' : 'bg-white text-zinc-900 shadow-sm') : 'text-zinc-500'}`}
              >
                Excerpts
              </button>
              <button
                onClick={() => onViewModeChange('full')}
                className={`px-3 py-1 rounded-md transition-all ${viewMode === 'full' ? (isDark ? 'bg-zinc-800 text-white shadow-sm' : 'bg-white text-zinc-900 shadow-sm') : 'text-zinc-500'}`}
              >
                Full Doc
              </button>
            </div> */}
          </div>
        </div>

        {/* Content Viewer */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-8">
          {/* EXCERPT VIEW DEACTIVATED - keeping code for potential future use */}
          {/* {viewMode === 'chunk' ? (
            <div className="space-y-8">
              <div className={`p-4 rounded-xl border flex items-start gap-3 ${isDark ? 'bg-zinc-900/50 border-zinc-800' : 'bg-white border-zinc-200 shadow-sm'}`}>
                <Info size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                <div className="text-xs leading-relaxed text-zinc-500">
                  Showing original source excerpts mentioned in the response.
                </div>
              </div>

              {currentDocCitations.map((citation, idx) => (
                <div key={idx} className="relative animate-in slide-in-from-left-2 duration-300" style={{ animationDelay: `${idx * 100}ms` }}>
                  <div className="absolute -left-4 top-0 bottom-0 w-1 bg-emerald-500/40 rounded-full" />
                  <div className={`italic text-sm leading-relaxed ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
                    "{citation.snippet}"
                  </div>
                </div>
              ))}

              {relevanceHighlights && (
                <div className={`p-4 rounded-xl border border-dashed flex items-start gap-3 animate-in fade-in
                  ${isDark ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-emerald-50 border-emerald-100'}`}>
                  <Sparkles size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                  <div className="text-xs leading-relaxed text-zinc-500">
                    AI highlighting ready. Switch to <span className="font-bold text-emerald-600 dark:text-emerald-400">Full Doc</span> to see semantic evidence in context.
                  </div>
                </div>
              )}
            </div>
          ) : ( */}
            <div className="min-w-0">
              <MarkdownRenderer
                content={doc?.content || ''}
                theme={theme}
                highlightTargets={allHighlightTargets}
              />
            </div>
          {/* )} */}
        </div>
      </div>
    </div>
  );
};

export default React.memo(SourceViewer);
