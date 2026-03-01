/**
 * @file MarkdownRenderer.tsx
 * @description Shared component for rendering Markdown content consistently across the app.
 * Handles theme-aware styling and verbatim text highlighting for RAG citations.
 */

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Theme } from '../types';

interface MarkdownRendererProps {
  content: string;
  theme: Theme;
  /** Verbatim strings that should be wrapped in <mark> tags */
  highlightTargets?: string[];
  className?: string;
}

/**
 * Internal helper to safely wrap verbatim strings in highlight tags 
 * without corrupting React's node tree.
 */
const HighlightedText: React.FC<{ 
  text: string; 
  allTargets: string[]; 
  isDark: boolean 
}> = ({ text, allTargets, isDark }) => {
  if (!text || typeof text !== 'string' || allTargets.length === 0) return <>{text}</>;

  let segments: (string | React.ReactNode)[] = [text];
  // Sort by length descending to match longest possible phrases first
  const sortedTargets = [...allTargets].sort((a, b) => b.length - a.length);

  sortedTargets.forEach((target) => {
    if (!target || target.trim().length < 3) return;
    const escaped = target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escaped})`, 'gi');

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (typeof seg === 'string') {
        const parts = seg.split(regex);
        if (parts.length > 1) {
          const newSegments: (string | React.ReactNode)[] = [];
          parts.forEach((part, idx) => {
            if (part === '') return;
            if (part.toLowerCase() === target.toLowerCase()) {
              newSegments.push(
                <mark 
                  key={`mark-${target}-${idx}`}
                  className={`px-0.5 rounded-sm transition-all duration-700 font-medium
                    ${isDark 
                      ? 'bg-emerald-500/30 text-emerald-50 ring-1 ring-emerald-500/20' 
                      : 'bg-emerald-100 text-emerald-900 ring-1 ring-emerald-200 shadow-sm'}`}
                >
                  {part}
                </mark>
              );
            } else {
              newSegments.push(part);
            }
          });
          segments.splice(i, 1, ...newSegments);
          i += newSegments.length - 1;
        }
      }
    }
  });

  return <>{segments}</>;
};

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  theme, 
  highlightTargets = [],
  className = "" 
}) => {
  const isDark = theme === 'dark';

  /**
   * Helper to apply highlighting to React children nodes
   */
  const withHighlights = (children: any) => {
    if (highlightTargets.length === 0) return children;
    return React.Children.map(children, child => {
      if (typeof child === 'string') {
        return <HighlightedText text={child} allTargets={highlightTargets} isDark={isDark} />;
      }
      return child;
    });
  };

  /**
   * Custom Markdown component mapping to ensure theme-aware colors 
   * for headers, bold text, and tables.
   */
  const components = useMemo(() => {
    const textColor = isDark ? 'text-zinc-100' : 'text-zinc-900';
    const borderColor = isDark ? 'border-zinc-800' : 'border-zinc-200';
    
    return {
      h1: ({ children }: any) => <h1 className={`text-xl font-bold mt-6 mb-4 ${textColor}`}>{withHighlights(children)}</h1>,
      h2: ({ children }: any) => <h2 className={`text-lg font-bold mt-5 mb-3 ${textColor}`}>{withHighlights(children)}</h2>,
      h3: ({ children }: any) => <h3 className={`text-base font-bold mt-4 mb-2 ${textColor}`}>{withHighlights(children)}</h3>,
      h4: ({ children }: any) => <h4 className={`text-sm font-bold mt-3 mb-1 ${textColor}`}>{withHighlights(children)}</h4>,
      strong: ({ children }: any) => <strong className={`font-bold ${textColor}`}>{withHighlights(children)}</strong>,
      p: ({ children }: any) => <p className="mb-4 leading-relaxed">{withHighlights(children)}</p>,
      li: ({ children }: any) => <li className="mb-1">{withHighlights(children)}</li>,
      em: ({ children }: any) => <em className="italic">{withHighlights(children)}</em>,
      table: ({ children }: any) => (
        <div className="overflow-x-auto my-6 rounded-xl border border-inherit shadow-sm">
          <table className={`min-w-full border-collapse ${borderColor}`}>
            {children}
          </table>
        </div>
      ),
      thead: ({ children }: any) => (
        <thead className={`${isDark ? 'bg-zinc-800/80' : 'bg-zinc-100'}`}>
          {children}
        </thead>
      ),
      th: ({ children }: any) => (
        <th className={`border px-4 py-3 text-left text-[11px] font-bold uppercase tracking-wider ${borderColor} ${isDark ? 'text-zinc-200' : 'text-zinc-600'}`}>
          {children}
        </th>
      ),
      td: ({ children }: any) => (
        <td className={`border px-4 py-3 text-sm leading-relaxed ${borderColor} ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
          {withHighlights(children)}
        </td>
      ),
      tr: ({ children }: any) => (
        <tr className={`transition-colors ${isDark ? 'odd:bg-zinc-900/50 even:bg-zinc-900 hover:bg-zinc-800/30' : 'odd:bg-white even:bg-zinc-50/50 hover:bg-zinc-100/50'}`}>
          {children}
        </tr>
      ),
      blockquote: ({ children }: any) => (
        <blockquote className={`border-l-4 pl-4 italic my-4 rounded-r-lg py-1 ${isDark ? 'border-emerald-900 bg-emerald-900/10 text-zinc-400' : 'border-emerald-200 bg-emerald-50 text-zinc-600'}`}>
          {children}
        </blockquote>
      )
    };
  }, [isDark, highlightTargets]);

  return (
    <div className={`prose prose-sm dark:prose-invert max-w-none prose-zinc ${isDark ? 'text-zinc-300' : 'text-zinc-800'} ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components as any}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default React.memo(MarkdownRenderer);
