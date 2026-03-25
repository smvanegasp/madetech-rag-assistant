/**
 * @file SourceViewer.tsx
 * @description Sliding side panel for inspecting document sources. 
 * Displays the full handbook document for any cited source.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, FileText, ChevronLeft, ChevronRight, Search, ChevronUp, ChevronDown } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { SelectedSource, ViewMode, Theme, HandbookDoc } from '../types';

interface SourceViewerProps {
  isOpen: boolean;
  onClose: () => void;
  source: SelectedSource | null;
  onViewModeChange: (mode: ViewMode) => void;
  onDocChange: (docId: string) => void;
  theme: Theme;
  handbookDocs: HandbookDoc[];
}

const SourceViewer: React.FC<SourceViewerProps> = ({
  isOpen,
  onClose,
  source,
  onViewModeChange,
  onDocChange,
  theme,
  handbookDocs
}) => {
  const { sources = [], currentDocId = '' } = source || {};

  const doc = handbookDocs.find(d => d.id === currentDocId) as HandbookDoc | undefined;
  const isDark = theme === 'dark';

  const distinctDocIds = Array.from(new Set(sources.map(s => s.docId)));
  const currentDocIndex = distinctDocIds.indexOf(currentDocId);

  const [searchQuery, setSearchQuery] = useState('');
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [totalMatches, setTotalMatches] = useState(0);

  const contentRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  /** Highlight the active match visually and scroll it into view. */
  const applyActiveHighlight = useCallback((index: number) => {
    if (!contentRef.current) return;
    const marks = contentRef.current.querySelectorAll<HTMLElement>('[data-search-match]');
    marks.forEach((el, i) => {
      if (i === index) {
        el.setAttribute('data-search-current', 'true');
        el.classList.add('!bg-emerald-500', '!text-white');
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      } else {
        el.removeAttribute('data-search-current');
        el.classList.remove('!bg-emerald-500', '!text-white');
      }
    });
  }, [isDark]);

  /** After each render with a new query or doc, recount matches and go to first. */
  useEffect(() => {
    if (!contentRef.current || !searchQuery.trim()) {
      setTotalMatches(0);
      setCurrentMatchIndex(0);
      return;
    }
    // Wait for the DOM to settle after the React render
    const id = setTimeout(() => {
      const marks = contentRef.current?.querySelectorAll('[data-search-match]');
      const count = marks?.length ?? 0;
      setTotalMatches(count);
      setCurrentMatchIndex(0);
      applyActiveHighlight(0);
    }, 50);
    return () => clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery, currentDocId]);

  const navigateMatch = useCallback((direction: 'prev' | 'next') => {
    if (totalMatches === 0) return;
    const next =
      direction === 'next'
        ? (currentMatchIndex + 1) % totalMatches
        : (currentMatchIndex - 1 + totalMatches) % totalMatches;
    setCurrentMatchIndex(next);
    applyActiveHighlight(next);
  }, [currentMatchIndex, totalMatches, applyActiveHighlight]);

  /** Ctrl+F / Cmd+F inside the panel focuses the search input. */
  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }
      if (e.key === 'Escape' && searchQuery) {
        setSearchQuery('');
      }
    };
    panel.addEventListener('keydown', handleKey);
    return () => panel.removeEventListener('keydown', handleKey);
  }, [searchQuery]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      navigateMatch(e.shiftKey ? 'prev' : 'next');
    }
  };

  if (!isOpen) return null;

  return (
    <div
      ref={panelRef}
      tabIndex={-1}
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

        {/* Toolbar: Document Navigation */}
        <div className={`px-4 py-2 flex items-center border-b text-xs ${isDark ? 'border-zinc-800 bg-zinc-900/30' : 'border-zinc-100 bg-zinc-50/50'}`}>
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
        </div>

        {/* Search Bar */}
        <div className={`px-3 py-2 flex items-center gap-2 border-b text-xs ${isDark ? 'border-zinc-800 bg-zinc-900/20' : 'border-zinc-100 bg-white'}`}>
          <Search size={13} className="shrink-0 text-zinc-400" />
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            onKeyDown={handleSearchKeyDown}
            placeholder="Find in document…"
            aria-label="Search document"
            className={`flex-1 bg-transparent outline-none text-xs placeholder:text-zinc-500
              ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}
          />
          {searchQuery && (
            <>
              <span className={`tabular-nums shrink-0 ${totalMatches === 0 ? 'text-red-400' : 'text-zinc-500'}`}>
                {totalMatches === 0 ? 'No results' : `${currentMatchIndex + 1} / ${totalMatches}`}
              </span>
              <button
                onClick={() => navigateMatch('prev')}
                disabled={totalMatches === 0}
                aria-label="Previous match"
                className="p-1 disabled:opacity-30 hover:bg-zinc-500/10 rounded transition-colors"
              >
                <ChevronUp size={13} />
              </button>
              <button
                onClick={() => navigateMatch('next')}
                disabled={totalMatches === 0}
                aria-label="Next match"
                className="p-1 disabled:opacity-30 hover:bg-zinc-500/10 rounded transition-colors"
              >
                <ChevronDown size={13} />
              </button>
              <button
                onClick={() => setSearchQuery('')}
                aria-label="Clear search"
                className="p-1 hover:bg-zinc-500/10 rounded transition-colors text-zinc-400 hover:text-zinc-600"
              >
                <X size={13} />
              </button>
            </>
          )}
        </div>

        {/* Content Viewer */}
        <div ref={contentRef} className="flex-1 overflow-y-auto p-6 sm:p-8">
          <div className="min-w-0">
            <MarkdownRenderer
              content={doc?.content || ''}
              theme={theme}
              searchQuery={searchQuery}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default React.memo(SourceViewer);
