/**
 * @file SourceViewer.tsx
 * @description Sliding side panel for inspecting document sources. 
 * Displays the full handbook document for any cited source.
 */

import React from 'react';
import { X, FileText, ChevronLeft, ChevronRight } from 'lucide-react';
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

  // Find the specific handbook document from backend data
  const doc = handbookDocs.find(d => d.id === currentDocId) as HandbookDoc | undefined;
  const isDark = theme === 'dark';

  // Navigation helpers for multi-document citations
  const distinctDocIds = Array.from(new Set(sources.map(s => s.docId)));
  const currentDocIndex = distinctDocIds.indexOf(currentDocId);

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

        {/* Content Viewer */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-8">
          <div className="min-w-0">
            <MarkdownRenderer
              content={doc?.content || ''}
              theme={theme}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default React.memo(SourceViewer);
