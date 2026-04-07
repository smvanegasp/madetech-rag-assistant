/**
 * @file App.tsx
 * @description The root orchestrator of the RAG Company Handbook application.
 * 
 * This component manages all global state and coordinates interactions between:
 * - ChatArea: Message display and input
 * - Sidebar: Chat history and navigation
 * - SourceViewer: Document inspection
 * 
 * State management:
 * - Conversations: Array of Chat objects with message history
 * - Theme: Dark/light mode synchronized with DOM
 * - UI Panels: Sidebar and SourceViewer visibility
 * 
 * Key patterns:
 * - Uses refs to prevent stale closures in async operations
 * - Manages unread notifications for background responses
 */

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SourceViewer from './components/SourceViewer';
import WelcomeModal from './components/WelcomeModal';
import ContactModal from './components/ContactModal';
import { Chat, Message, SourceChunk, ToolStep, SelectedSource, ViewMode, Theme, UserProfile, HandbookDoc } from './types';
import { getHandbookResponseStreamed, getHandbookDocs } from './services/apiService';
import { PanelRight, MessageCircleHeart } from 'lucide-react';

const App: React.FC = () => {
  // --- UI STATE ---
  const [theme, setTheme] = useState<Theme>('light');
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [sourceViewerOpen, setSourceViewerOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState<SelectedSource | null>(null);
  const [showWelcome, setShowWelcome] = useState<boolean>(
    () => !localStorage.getItem('welcomeSeen')
  );
  const [showContact, setShowContact] = useState(false);

  const handleCloseWelcome = useCallback(() => {
    localStorage.setItem('welcomeSeen', '1');
    setShowWelcome(false);
  }, []);

  // --- DATA STATE ---
  // --- CONVERSATION STATE ---
  /** Array of all chat conversations */
  const [chats, setChats] = useState<Chat[]>([]);
  /** ID of currently active chat */
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  /** Current value of message input field */
  const [inputValue, setInputValue] = useState('');
  /** Live tool steps shown during loading (from SSE stream) */
  const [liveToolSteps, setLiveToolSteps] = useState<ToolStep[]>([]);
  /** All loaded handbook documents (fetched from backend on mount) */
  const [handbookDocs, setHandbookDocs] = useState<HandbookDoc[]>([]);
  
  /**
   * Keeps track of the active chat ID in a ref to prevent stale closures.
   * This is crucial for async operations (like AI responses) that need to
   * know which chat was active when they were initiated.
   */
  const currentChatIdRef = useRef<string | null>(null);
  
  useEffect(() => {
    currentChatIdRef.current = currentChatId;
    setLiveToolSteps([]);
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
    setLiveToolSteps([]);

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
  const MAX_ATTEMPTS = 3;

  const handleSend = async (retryQuery?: string) => {
    const isRetry = typeof retryQuery === 'string' && retryQuery.length > 0;
    const query = isRetry ? retryQuery : inputValue.trim();
    if (!query || !currentChatId) return;

    const userQuery = query;
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
          messages: [...chat.messages.filter(m => !m.isError), userMessage],
          title: newTitle,
          isLoading: true
        };
      }
      return chat;
    }));

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        setLiveToolSteps([]);
        const response = await getHandbookResponseStreamed(
          userQuery,
          currentChat?.messages || [],
          currentChat?.dbChatId,
          (step) => {
            if (currentChatIdRef.current !== activeChatId) return;
            const key = `${step.tool_name}:${JSON.stringify(step.arguments)}`;
            setLiveToolSteps(prev => {
              if (prev.some(s => `${s.tool_name}:${JSON.stringify(s.arguments)}` === key)) return prev;
              return [...prev, step];
            });
          },
        );

        if (!response.isError) {
          const assistantMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: response.content,
            sources: response.sources,
            toolSteps: response.tool_steps ?? undefined,
            timestamp: new Date(),
            isError: false,
          };

          setLiveToolSteps([]);
          setChats(prev => prev.map(chat => {
            if (chat.id === activeChatId) {
              const isBackground = currentChatIdRef.current !== activeChatId;
              return {
                ...chat,
                messages: [...chat.messages, assistantMessage],
                updatedAt: new Date(),
                isLoading: false,
                hasUnreadResponse: isBackground,
                dbChatId: chat.dbChatId ?? response.chat_id ?? undefined,
              };
            }
            return chat;
          }));
          return;
        }

        if (attempt === MAX_ATTEMPTS) {
          const errorMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: response.content || "I'm having trouble connecting right now. Please try again.",
            timestamp: new Date(),
            isError: true,
          };
          setLiveToolSteps([]);
          setChats(prev => prev.map(chat =>
            chat.id === activeChatId ? { ...chat, messages: [...chat.messages, errorMessage], isLoading: false } : chat
          ));
        }
      } catch (err) {
        if (attempt === MAX_ATTEMPTS) {
          console.error("All retry attempts failed:", err);
          const errorMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: "I'm having trouble connecting right now. Please check your connection and try again.",
            timestamp: new Date(),
            isError: true,
          };
          setLiveToolSteps([]);
          setChats(prev => prev.map(chat =>
            chat.id === activeChatId ? { ...chat, messages: [...chat.messages, errorMessage], isLoading: false } : chat
          ));
        }
      }
    }
  };

  const handleRetry = useCallback(async () => {
    if (!currentChatId) return;
    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;
    const lastUserMsg = [...chat.messages].reverse().find(m => m.role === 'user');
    if (!lastUserMsg) return;
    const activeChatId = currentChatId;
    const messagesWithoutError = chat.messages.filter(m => !m.isError);

    setChats(prev => prev.map(c =>
      c.id === activeChatId ? { ...c, messages: messagesWithoutError, isLoading: true } : c
    ));

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        setLiveToolSteps([]);
        const response = await getHandbookResponseStreamed(
          lastUserMsg.content,
          messagesWithoutError,
          chat.dbChatId,
          (step) => {
            if (currentChatIdRef.current !== activeChatId) return;
            const key = `${step.tool_name}:${JSON.stringify(step.arguments)}`;
            setLiveToolSteps(prev => {
              if (prev.some(s => `${s.tool_name}:${JSON.stringify(s.arguments)}` === key)) return prev;
              return [...prev, step];
            });
          },
        );

        if (!response.isError) {
          const assistantMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: response.content,
            sources: response.sources,
            toolSteps: response.tool_steps ?? undefined,
            timestamp: new Date(),
            isError: false,
          };
          setLiveToolSteps([]);
          setChats(prev => prev.map(c => {
            if (c.id === activeChatId) {
              const isBackground = currentChatIdRef.current !== activeChatId;
              return {
                ...c,
                messages: [...c.messages.filter(m => !m.isError), assistantMessage],
                updatedAt: new Date(),
                isLoading: false,
                hasUnreadResponse: isBackground,
                dbChatId: c.dbChatId ?? response.chat_id ?? undefined,
              };
            }
            return c;
          }));
          return;
        }

        if (attempt === MAX_ATTEMPTS) {
          const errorMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: response.content || "I'm having trouble connecting right now. Please try again.",
            timestamp: new Date(),
            isError: true,
          };
          setLiveToolSteps([]);
          setChats(prev => prev.map(c =>
            c.id === activeChatId ? { ...c, messages: [...c.messages.filter(m => !m.isError), errorMessage], isLoading: false } : c
          ));
        }
      } catch (err) {
        if (attempt === MAX_ATTEMPTS) {
          console.error("All retry attempts failed:", err);
          const errorMessage: Message = {
            id: Math.random().toString(36).substring(7),
            role: 'assistant',
            content: "I'm having trouble connecting right now. Please check your connection and try again.",
            timestamp: new Date(),
            isError: true,
          };
          setLiveToolSteps([]);
          setChats(prev => prev.map(c =>
            c.id === activeChatId ? { ...c, messages: [...c.messages.filter(m => !m.isError), errorMessage], isLoading: false } : c
          ));
        }
      }
    }
  }, [currentChatId, chats]);

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
    <div className={`flex h-dvh w-full overflow-hidden transition-all duration-300 ${isDark ? 'bg-zinc-950 text-zinc-200' : 'bg-white text-zinc-900'}`}>
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
              {currentChat?.title || 'Nexus'}
            </div>
          </div>
          {/* Feedback button — header on mobile, hidden on desktop (floating button used instead) */}
          <div className="flex items-center xl:hidden">
            <button
              onClick={() => setShowContact(true)}
              aria-label="Feedback & Contact"
              className={`p-2 rounded-lg transition-all ${isDark ? 'hover:bg-zinc-800 text-zinc-400' : 'hover:bg-zinc-100 text-zinc-500'}`}
            >
              <MessageCircleHeart size={18} />
            </button>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          <ChatArea
            messages={currentChat?.messages || []}
            inputValue={inputValue}
            setInputValue={setInputValue}
            onSend={handleSend}
            onRetry={handleRetry}
            isLoading={currentChat?.isLoading || false}
            liveToolSteps={liveToolSteps}
            onOpenSource={handleOpenSource}
            theme={theme}
            handbookDocs={handbookDocs}
            onOpenDisclaimer={() => setShowWelcome(true)}
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
          handbookDocs={handbookDocs}
        />
      </main>

      {sidebarOpen && window.innerWidth < 768 && (
        <div 
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[45]"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Floating feedback button — desktop only, mobile uses header button instead */}
      <button
        onClick={() => setShowContact(true)}
        aria-label="Feedback & Contact"
        className="hidden xl:flex fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full bg-emerald-500 hover:bg-emerald-600 active:scale-95
          text-white shadow-lg hover:shadow-emerald-500/30 hover:shadow-xl transition-all duration-200 items-center justify-center"
      >
        <MessageCircleHeart size={20} />
      </button>

      <WelcomeModal
        isOpen={showWelcome}
        onClose={handleCloseWelcome}
        theme={theme}
      />

      <ContactModal
        isOpen={showContact}
        onClose={() => setShowContact(false)}
        theme={theme}
      />
    </div>
  );
};

export default App;