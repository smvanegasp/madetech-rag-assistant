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
  timestamp: Date;
  isError?: boolean;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
  isLoading?: boolean;
  hasUnreadResponse?: boolean;
  isCustomTitle?: boolean;
  /** Backend-assigned UUID that groups all interactions in this conversation */
  dbChatId?: string;
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
