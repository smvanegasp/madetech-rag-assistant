/**
 * @file MarkdownRenderer.tsx
 * @description Shared component for rendering Markdown content consistently across the app.
 * Handles theme-aware styling for headers, bold text, and tables.
 */

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Theme, SourceChunk } from '../types';

interface MarkdownRendererProps {
  content: string;
  theme: Theme;
  className?: string;
  searchQuery?: string;
  sources?: SourceChunk[];
  onCiteClick?: (docId: string) => void;
}

/**
 * Recursively walks React children and wraps text nodes that match the search
 * query in <mark> elements. Non-matching text and non-string nodes are
 * returned unchanged.
 */
function highlightText(
  children: React.ReactNode,
  query: string,
  isDark: boolean,
  keyPrefix = 'hl'
): React.ReactNode {
  if (!query) return children;

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const markClass = isDark
    ? 'bg-emerald-500/25 text-emerald-200 rounded-sm px-0.5'
    : 'bg-emerald-100 text-emerald-900 rounded-sm px-0.5';

  const processNode = (node: React.ReactNode, idx: number | string): React.ReactNode => {
    if (typeof node === 'string') {
      const parts = node.split(regex);
      if (parts.length === 1) return node;
      return parts.map((part, i) =>
        regex.test(part) ? (
          <mark
            key={`${keyPrefix}-${idx}-${i}`}
            data-search-match="true"
            className={markClass}
          >
            {part}
          </mark>
        ) : (
          part
        )
      );
    }
    if (React.isValidElement(node)) {
      const el = node as React.ReactElement<{ children?: React.ReactNode }>;
      if (el.props.children) {
        return React.cloneElement(el, {
          children: highlightText(el.props.children, query, isDark, `${keyPrefix}-${idx}`),
        } as any);
      }
    }
    if (Array.isArray(node)) {
      return node.map((child, i) => processNode(child, `${keyPrefix}-arr-${i}`));
    }
    return node;
  };

  if (Array.isArray(children)) {
    return children.map((child, i) => processNode(child, i));
  }
  return processNode(children, 0);
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  theme,
  className = "",
  searchQuery = "",
  sources,
  onCiteClick
}) => {
  const isDark = theme === 'dark';
  const trimmedQuery = searchQuery.trim();

  const hl = (children: React.ReactNode) =>
    trimmedQuery ? highlightText(children, trimmedQuery, isDark) : children;

  /** Strip [n] citation markers from text so they don't show inline */
  const stripCitations = (children: React.ReactNode): React.ReactNode => {
    if (!sources) return children;
    const processNode = (node: React.ReactNode): React.ReactNode => {
      if (typeof node === 'string') return node.replace(/\s*(?:\[\d+\]|【\d+】)/g, '');
      if (React.isValidElement(node)) {
        const el = node as React.ReactElement<{ children?: React.ReactNode }>;
        if (el.props.children) {
          return React.cloneElement(el, { children: stripCitations(el.props.children) } as any);
        }
      }
      if (Array.isArray(node)) return node.map(processNode);
      return node;
    };
    if (Array.isArray(children)) return children.map(processNode);
    return processNode(children);
  };

  const process = (children: React.ReactNode) => stripCitations(hl(children));

  /**
   * Custom Markdown component mapping to ensure theme-aware colors 
   * for headers, bold text, and tables.
   */
  const components = useMemo(() => {
    const textColor = isDark ? 'text-zinc-100' : 'text-zinc-900';
    const borderColor = isDark ? 'border-zinc-800' : 'border-zinc-200';
    
    return {
      h1: ({ children }: any) => <h1 className={`text-xl font-bold mt-6 mb-4 ${textColor}`}>{process(children)}</h1>,
      h2: ({ children }: any) => <h2 className={`text-lg font-bold mt-5 mb-3 ${textColor}`}>{process(children)}</h2>,
      h3: ({ children }: any) => <h3 className={`text-base font-bold mt-4 mb-2 ${textColor}`}>{process(children)}</h3>,
      h4: ({ children }: any) => <h4 className={`text-sm font-bold mt-3 mb-1 ${textColor}`}>{process(children)}</h4>,
      strong: ({ children }: any) => <strong className={`font-bold ${textColor}`}>{process(children)}</strong>,
      p: ({ children }: any) => <p className="mb-4 leading-relaxed">{process(children)}</p>,
      li: ({ children }: any) => <li className="mb-1">{process(children)}</li>,
      em: ({ children }: any) => <em className="italic">{process(children)}</em>,
      table: ({ children }: any) => (
        <div className="overflow-x-auto max-w-full my-6 rounded-xl border border-inherit shadow-sm">
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
          {process(children)}
        </th>
      ),
      td: ({ children }: any) => (
        <td className={`border px-4 py-3 text-sm leading-relaxed ${borderColor} ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
          {process(children)}
        </td>
      ),
      tr: ({ children }: any) => (
        <tr className={`transition-colors ${isDark ? 'odd:bg-zinc-900/50 even:bg-zinc-900 hover:bg-zinc-800/30' : 'odd:bg-white even:bg-zinc-50/50 hover:bg-zinc-100/50'}`}>
          {children}
        </tr>
      ),
      blockquote: ({ children }: any) => (
        <blockquote className={`border-l-4 pl-4 italic my-4 py-1 ${isDark ? 'border-zinc-700 text-zinc-400' : 'border-zinc-300 text-zinc-600'}`}>
          {process(children)}
        </blockquote>
      ),
      a: ({ href, children }: any) => (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={`font-medium underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-700'}`}
        >
          {process(children)}
        </a>
      )
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDark, trimmedQuery, sources]);

  return (
    <div className={`prose prose-sm dark:prose-invert max-w-none prose-zinc ${isDark ? 'text-zinc-300' : 'text-zinc-800'} ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={components as any}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default React.memo(MarkdownRenderer);
