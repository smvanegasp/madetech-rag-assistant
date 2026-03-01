/**
 * @file App.tsx
 * @description The root orchestrator of the RAG Company Handbook application.
 * 
 * This component manages all global state and coordinates interactions between:
 * - ChatArea: Message display and input
 * - Sidebar: Chat history and navigation
 * - SourceViewer: Document inspection with highlighting
 * 
 * State management:
 * - Conversations: Array of Chat objects with message history
 * - Theme: Dark/light mode synchronized with DOM
 * - UI Panels: Sidebar and SourceViewer visibility
 * - Analysis Tasks: Tracks document highlighting operations
 * 
 * Key patterns:
 * - Uses refs to prevent stale closures in async operations
 * - Tracks per-document analysis tasks with Set<"messageId|docId">
 * - Manages unread notifications for background responses
 */

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SourceViewer from './components/SourceViewer';
import { Chat, Message, SourceChunk, SelectedSource, ViewMode, Theme, UserProfile, HandbookDoc } from './types';
import { getHandbookResponse, getRelevanceHighlights, getHandbookDocs } from './services/apiService';
import { PanelRight, Bell, BellRing } from 'lucide-react';

const App: React.FC = () => {
  // --- UI STATE ---
  const [theme, setTheme] = useState<Theme>('light');
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [sourceViewerOpen, setSourceViewerOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState<SelectedSource | null>(null);
  const [showNotifications, setShowNotifications] = useState(false);

  // --- DATA STATE ---
  // --- CONVERSATION STATE ---
  /** Array of all chat conversations */
  const [chats, setChats] = useState<Chat[]>([]);
  /** ID of currently active chat */
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  /** Current value of message input field */
  const [inputValue, setInputValue] = useState('');
  /** All loaded handbook documents (fetched from backend on mount) */
  const [handbookDocs, setHandbookDocs] = useState<HandbookDoc[]>([]);
  
  /** 
   * Tracks specific document analysis tasks currently in progress.
   * Each task is identified by "messageId|docId" format.
   * This allows multiple documents to be analyzed simultaneously.
   */
  const [analyzingTasks, setAnalyzingTasks] = useState<Set<string>>(new Set());

  /**
   * Keeps track of the active chat ID in a ref to prevent stale closures.
   * This is crucial for async operations (like AI responses) that need to
   * know which chat was active when they were initiated.
   */
  const currentChatIdRef = useRef<string | null>(null);
  
  useEffect(() => {
    currentChatIdRef.current = currentChatId;
    if (currentChatId) {
      setChats(prev => prev.map(c => 
        c.id === currentChatId ? { ...c, hasUnreadResponse: false } : c
      ));
    }
  }, [currentChatId]);

  const profile = useMemo<UserProfile>(() => ({
    name: 'Sergio Vanegas',
    role: 'Product Manager',
    avatarUrl: 'https://api.dicebear.com/7.x/bottts/svg?seed=Sergio&baseColor=10b981'
  }), []);

  const toggleTheme = useCallback(() => setTheme(prev => prev === 'light' ? 'dark' : 'light'), []);

  useEffect(() => {
    const isDark = theme === 'dark';
    document.documentElement.classList.toggle('dark', isDark);
    document.body.style.backgroundColor = isDark ? '#09090b' : '#ffffff'; 
  }, [theme]);

  const handleNewChat = useCallback(() => {
    const id = Math.random().toString(36).substring(7);
    const newChat: Chat = {
      id,
      title: 'New Chat',
      messages: [],
      updatedAt: new Date(),
      isLoading: false,
      hasUnreadResponse: false,
      isCustomTitle: false
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(id);
    if (window.innerWidth < 768) setSidebarOpen(false);
  }, []);

  // Load handbook documents from backend on mount
  useEffect(() => {
    const loadHandbooks = async () => {
      const docs = await getHandbookDocs();
      setHandbookDocs(docs);
    };
    loadHandbooks();
  }, []);

  useEffect(() => {
    if (chats.length === 0) {
      handleNewChat();
    }
    const handleResize = () => {
      if (window.innerWidth < 768) setSidebarOpen(false);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [chats.length, handleNewChat]);

  const currentChat = chats.find(c => c.id === currentChatId);
  const unreadChats = chats.filter(c => c.hasUnreadResponse);

  /**
   * Updates the highlights for a specific document in a specific message.
   * 
   * Highlights are stored per-document to allow incremental analysis. When a user
   * requests highlights for a document, we store them in the message state so they
   * persist even if the user switches to viewing a different document.
   * 
   * @param chatId - ID of the chat containing the message
   * @param messageId - ID of the message to update
   * @param docId - ID of the document that was analyzed
   * @param highlights - Array of verbatim strings to highlight in the document
   */
  const updateMessageHighlights = useCallback((chatId: string, messageId: string, docId: string, highlights: string[]) => {
    setChats(prev => prev.map(c => {
      if (c.id !== chatId) return c;
      return {
        ...c,
        messages: c.messages.map(m => {
          if (m.id !== messageId) return m;
          return {
            ...m,
            highlights: {
              ...(m.highlights || {}),
              [docId]: highlights
            }
          };
        })
      };
    }));
  }, []);

  /**
   * Triggers AI-powered semantic analysis for a specific document.
   * 
   * This function powers the "Highlight with AI" feature. When a user views a
   * source document in the SourceViewer, they can click to highlight which specific
   * phrases support the assistant's answer.
   * 
   * Process:
   * 1. Mark the document analysis task as in-progress (shows loading spinner)
   * 2. Call backend /api/highlights with answer and document content
   * 3. Receive array of verbatim strings to highlight
   * 4. Store highlights in message state (persists even if user switches documents)
   * 5. Remove task from analyzing set (hides loading spinner)
   * 
   * The task key format "messageId|docId" ensures:
   * - Multiple documents can be analyzed simultaneously
   * - Each document is only analyzed once (prevents duplicate calls)
   * - Analysis persists across document switches
   * 
   * @param messageId - ID of the assistant message whose answer we're verifying
   * @param docId - ID of the document to analyze
   * @param answer - The assistant's answer text (what we're looking for support for)
   */
  const handleTriggerDocAnalysis = useCallback(async (messageId: string, docId: string, answer: string) => {
    const taskKey = `${messageId}|${docId}`;
    if (analyzingTasks.has(taskKey) || !currentChatId) return;

    // Mark specific document task as analyzing
    setAnalyzingTasks(prev => new Set(prev).add(taskKey));

    const activeChatId = currentChatId;
    const doc = handbookDocs.find(d => d.id === docId);
    
    if (!doc) {
      setAnalyzingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskKey);
        return next;
      });
      return;
    }

    try {
      const highlights = await getRelevanceHighlights(answer, doc.content);
      updateMessageHighlights(activeChatId, messageId, docId, highlights);
    } catch (e) {
      console.error(`Analysis failed for doc ${docId}:`, e);
    } finally {
      // Cleanup task state
      setAnalyzingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskKey);
        return next;
      });
    }
  }, [analyzingTasks, currentChatId, handbookDocs, updateMessageHighlights]);

  /**
   * Handles sending a user message and receiving the AI response.
   * 
   * This is the core message flow that:
   * 1. Creates a user message object
   * 2. Updates chat title (from first message if not custom)
   * 3. Sets loading state
   * 4. Calls backend /api/chat with query and history
   * 5. Creates assistant message with response and sources
   * 6. Marks chat as unread if user switched to a different chat
   * 
   * The unread notification system tracks responses that arrive while the user
   * is viewing a different chat, allowing them to return later.
   */
  const handleSend = async () => {
    if (!inputValue.trim() || !currentChatId) return;

    const userQuery = inputValue.trim();
    const activeChatId = currentChatId;
    setInputValue('');

    const userMessage: Message = {
      id: Math.random().toString(36).substring(7),
      role: 'user',
      content: userQuery,
      timestamp: new Date()
    };

    setChats(prev => prev.map(chat => {
      if (chat.id === activeChatId) {
        const shouldGenerateTitle = chat.messages.length === 0 && !chat.isCustomTitle;
        const newTitle = shouldGenerateTitle 
          ? userQuery.substring(0, 30) + (userQuery.length > 30 ? '...' : '') 
          : chat.title;

        return { 
          ...chat, 
          messages: [...chat.messages, userMessage],
          title: newTitle,
          isLoading: true
        };
      }
      return chat;
    }));

    try {
      const response = await getHandbookResponse(userQuery, currentChat?.messages || []);
      const assistantMessageId = Math.random().toString(36).substring(7);
      
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: response.content,
        sources: response.sources,
        timestamp: new Date(),
        highlights: {}
      };

      setChats(prev => prev.map(chat => {
          if (chat.id === activeChatId) {
            const isBackground = currentChatIdRef.current !== activeChatId;
            return { 
              ...chat, 
              messages: [...chat.messages, assistantMessage], 
              updatedAt: new Date(),
              isLoading: false,
              hasUnreadResponse: isBackground
            };
          }
          return chat;
      }));

    } catch (err) {
      console.error("Critical AI Communication Error:", err);
      setChats(prev => prev.map(chat => 
        chat.id === activeChatId ? { ...chat, isLoading: false } : chat
      ));
    }
  };

  const handleDeleteChat = useCallback((id: string) => {
    setChats(prev => prev.filter(c => c.id !== id));
  }, []);

  const handleRenameChat = useCallback((id: string, newTitle: string) => {
    setChats(prev => prev.map(c => 
      c.id === id ? { ...c, title: newTitle, isCustomTitle: true } : c
    ));
  }, []);

  const handleOpenSource = useCallback((sources: SourceChunk[], docId: string, messageId: string) => {
    setSelectedSource({
      sources,
      currentDocId: docId,
      viewMode: 'full', // Default to full document view
      contextMessageId: messageId
    });
    setSourceViewerOpen(true);
  }, []);

  const handleDocChange = useCallback((docId: string) => {
    setSelectedSource(prev => prev ? { ...prev, currentDocId: docId } : null);
  }, []);

  const handleViewModeChange = useCallback((mode: ViewMode) => {
    setSelectedSource(prev => prev ? { ...prev, viewMode: mode } : null);
  }, []);

  const isDark = theme === 'dark';

  return (
    <div className={`flex h-screen w-full overflow-hidden transition-all duration-300 ${isDark ? 'bg-zinc-950 text-zinc-200' : 'bg-white text-zinc-900'}`}>
      <Sidebar 
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={setCurrentChatId}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={handleRenameChat}
        theme={theme}
        toggleTheme={toggleTheme}
        profile={profile}
      />

      <main className="flex-1 flex flex-col h-full relative min-w-0">
        <header className={`h-14 flex items-center px-4 shrink-0 z-30 transition-colors
          ${isDark ? 'bg-zinc-900/50 backdrop-blur border-b border-zinc-800' : 'bg-white/80 backdrop-blur border-b border-zinc-100'}`}>
          <div className="flex items-center gap-2">
            {!sidebarOpen && (
              <button 
                onClick={() => setSidebarOpen(true)}
                aria-label="Expand sidebar"
                className={`p-2 rounded-lg transition-all ${isDark ? 'hover:bg-zinc-800 text-zinc-400' : 'hover:bg-zinc-100 text-zinc-500'}`}
              >
                <PanelRight size={18} />
              </button>
            )}
          </div>
          <div className="flex-1 flex justify-center px-4 sm:px-8">
            <div className={`text-sm font-semibold tracking-tight ${isDark ? 'text-zinc-400' : 'text-zinc-500'} truncate max-w-[150px] sm:max-w-md`}>
              {currentChat?.title || 'Nexus AI'}
            </div>
          </div>
          <div className="relative flex items-center gap-2">
            <button 
              onClick={() => setShowNotifications(!showNotifications)}
              aria-label="Activity Feed"
              className={`p-2 rounded-lg transition-all relative ${isDark ? 'hover:bg-zinc-800 text-zinc-400' : 'hover:bg-zinc-100 text-zinc-500'}`}
            >
              {unreadChats.length > 0 ? (
                <BellRing size={18} className="text-emerald-500 animate-pulse" />
              ) : (
                <Bell size={18} />
              )}
              {unreadChats.length > 0 && (
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-emerald-500 rounded-full border border-white dark:border-zinc-900"></span>
              )}
            </button>
            {showNotifications && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowNotifications(false)} />
                <div className={`absolute right-0 top-full mt-2 w-64 rounded-xl border shadow-xl z-50 p-2 animate-in slide-in-from-top-2 duration-200
                  ${isDark ? 'bg-zinc-900 border-zinc-800' : 'bg-white border-zinc-200'}`}>
                  <div className="px-3 py-2 text-xs font-bold text-zinc-500 uppercase tracking-wider">Activity Feed</div>
                  <div className="space-y-1 max-h-60 overflow-y-auto">
                    {unreadChats.length === 0 ? (
                      <div className="px-3 py-4 text-xs text-zinc-400 text-center italic">No new activity.</div>
                    ) : (
                      unreadChats.map(c => (
                        <button key={c.id} onClick={() => { setCurrentChatId(c.id); setShowNotifications(false); }} className={`w-full text-left p-3 rounded-lg flex items-center gap-3 transition-colors ${isDark ? 'hover:bg-zinc-800' : 'hover:bg-zinc-50'}`}>
                          <div className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="text-xs font-medium truncate">{c.title}</div>
                            <div className="text-[10px] text-zinc-500">New response waiting</div>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          <ChatArea 
            messages={currentChat?.messages || []}
            inputValue={inputValue}
            setInputValue={setInputValue}
            onSend={handleSend}
            isLoading={currentChat?.isLoading || false}
            onOpenSource={handleOpenSource}
            theme={theme}
            handbookDocs={handbookDocs}
          />
        </div>

        {sourceViewerOpen && (
          <div 
            className="fixed inset-0 bg-black/20 backdrop-blur-[1px] z-[55] animate-in fade-in duration-300"
            onClick={() => setSourceViewerOpen(false)}
          />
        )}

        <SourceViewer 
          isOpen={sourceViewerOpen}
          onClose={() => setSourceViewerOpen(false)}
          source={selectedSource}
          onViewModeChange={handleViewModeChange}
          onDocChange={handleDocChange}
          theme={theme}
          activeChat={currentChat}
          isDocAnalyzing={analyzingTasks.has(`${selectedSource?.contextMessageId}|${selectedSource?.currentDocId}`)}
          onTriggerAnalysis={() => {
            const msg = currentChat?.messages.find(m => m.id === selectedSource?.contextMessageId);
            if (msg && selectedSource) {
              handleTriggerDocAnalysis(msg.id, selectedSource.currentDocId, msg.content);
            }
          }}
          handbookDocs={handbookDocs}
        />
      </main>

      {sidebarOpen && window.innerWidth < 768 && (
        <div 
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[45]"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

export default App;