/**
 * @file MarkdownRenderer.tsx
 * @description Shared component for rendering Markdown content consistently across the app.
 * Handles theme-aware styling for headers, bold text, and tables.
 */

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Theme } from '../types';

interface MarkdownRendererProps {
  content: string;
  theme: Theme;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  theme, 
  className = "" 
}) => {
  const isDark = theme === 'dark';

  /**
   * Custom Markdown component mapping to ensure theme-aware colors 
   * for headers, bold text, and tables.
   */
  const components = useMemo(() => {
    const textColor = isDark ? 'text-zinc-100' : 'text-zinc-900';
    const borderColor = isDark ? 'border-zinc-800' : 'border-zinc-200';
    
    return {
      h1: ({ children }: any) => <h1 className={`text-xl font-bold mt-6 mb-4 ${textColor}`}>{children}</h1>,
      h2: ({ children }: any) => <h2 className={`text-lg font-bold mt-5 mb-3 ${textColor}`}>{children}</h2>,
      h3: ({ children }: any) => <h3 className={`text-base font-bold mt-4 mb-2 ${textColor}`}>{children}</h3>,
      h4: ({ children }: any) => <h4 className={`text-sm font-bold mt-3 mb-1 ${textColor}`}>{children}</h4>,
      strong: ({ children }: any) => <strong className={`font-bold ${textColor}`}>{children}</strong>,
      p: ({ children }: any) => <p className="mb-4 leading-relaxed">{children}</p>,
      li: ({ children }: any) => <li className="mb-1">{children}</li>,
      em: ({ children }: any) => <em className="italic">{children}</em>,
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
          {children}
        </td>
      ),
      tr: ({ children }: any) => (
        <tr className={`transition-colors ${isDark ? 'odd:bg-zinc-900/50 even:bg-zinc-900 hover:bg-zinc-800/30' : 'odd:bg-white even:bg-zinc-50/50 hover:bg-zinc-100/50'}`}>
          {children}
        </tr>
      ),
      blockquote: ({ children }: any) => (
        <blockquote className={`border-l-4 pl-4 italic my-4 py-1 ${isDark ? 'border-zinc-700 text-zinc-400' : 'border-zinc-300 text-zinc-600'}`}>
          {children}
        </blockquote>
      ),
      a: ({ href, children }: any) => (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={`font-medium underline underline-offset-2 transition-colors ${isDark ? 'text-emerald-400 hover:text-emerald-300' : 'text-emerald-600 hover:text-emerald-700'}`}
        >
          {children}
        </a>
      )
    };
  }, [isDark]);

  return (
    <div className={`prose prose-sm dark:prose-invert max-w-none prose-zinc ${isDark ? 'text-zinc-300' : 'text-zinc-800'} ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={components as any}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default React.memo(MarkdownRenderer);
