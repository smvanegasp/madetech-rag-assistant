/**
 * @file types.ts
 * @description Centralized TypeScript definitions for the Handbook Assistant.
 * Defines the shape of our knowledge base, messaging system, and UI state.
 */

export interface HandbookDoc {
  id: string;
  title: string;
  category: string;
  content: string;
}

export interface SourceChunk {
  docId: string;
  snippet: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  /** 
   * highlights: Map of docId to an array of verbatim strings 
   * found to be relevant to this specific answer. 
   */
  highlights?: Record<string, string[]>;
  timestamp: Date;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
  isLoading?: boolean;
  hasUnreadResponse?: boolean;
  isCustomTitle?: boolean;
}

/** Determines if the Source Viewer shows just fragments or the whole doc */
export type ViewMode = 'chunk' | 'full';

export interface SelectedSource {
  sources: SourceChunk[]; 
  currentDocId: string;   
  viewMode: ViewMode;
  contextMessageId: string; 
}

export type Theme = 'light' | 'dark';

export interface UserProfile {
  name: string;
  role: string;
  avatarUrl: string;
}
