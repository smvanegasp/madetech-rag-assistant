/**
 * @file Sidebar.tsx
 * @description Navigation component managing chat history, settings, and user profile.
 * Features inline renaming and destructive deletion of conversation threads.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  MessageSquare, 
  Plus, 
  Sun, 
  Moon,
  PanelLeft,
  Loader2,
  Trash2,
  MoreHorizontal,
  Pencil
} from 'lucide-react';
import { Chat, Theme, UserProfile } from '../types';

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  chats: Chat[];
  currentChatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onRenameChat: (id: string, newTitle: string) => void;
  theme: Theme;
  toggleTheme: () => void;
  profile: UserProfile;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  isOpen, 
  setIsOpen, 
  chats, 
  currentChatId, 
  onSelectChat, 
  onNewChat,
  onDeleteChat,
  onRenameChat,
  theme,
  toggleTheme,
  profile
}) => {
  const isDark = theme === 'dark';
  
  // Local UI states for editing chat metadata
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  
  const editInputRef = useRef<HTMLInputElement>(null);

  // Focus input automatically when entering rename mode
  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
    }
  }, [editingId]);

  const handleStartRename = useCallback((chat: Chat) => {
    setEditingId(chat.id);
    setEditTitle(chat.title);
    setOpenMenuId(null);
  }, []);

  const handleSaveRename = useCallback((id: string) => {
    if (editTitle.trim()) {
      onRenameChat(id, editTitle.trim());
    }
    setEditingId(null);
  }, [editTitle, onRenameChat]);

  const handleCancelRename = useCallback(() => {
    setEditingId(null);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent, id: string) => {
    if (e.key === 'Enter') handleSaveRename(id);
    if (e.key === 'Escape') handleCancelRename();
  };

  return (
    <aside 
      className={`h-full transition-all duration-300 flex flex-col z-[50]
        ${isDark ? 'bg-zinc-950 border-r border-zinc-800 text-zinc-200' : 'bg-zinc-50 border-r border-zinc-200 text-zinc-900'}
        ${isOpen 
          ? 'fixed inset-y-0 left-0 w-64 sm:relative translate-x-0' 
          : 'fixed inset-y-0 left-0 w-64 -translate-x-full sm:relative sm:translate-x-0 sm:w-0 overflow-hidden'
        }`}
      aria-hidden={!isOpen}
    >
      <div className="flex flex-col h-full">
        {/* New Conversation Button */}
        <div className="p-4 flex items-center justify-between">
          <button 
            onClick={onNewChat}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all flex-1 text-sm font-medium
              ${isDark ? 'hover:bg-zinc-900 text-zinc-200' : 'hover:bg-zinc-200 text-zinc-700'}`}
          >
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${isDark ? 'bg-zinc-800' : 'bg-white shadow-sm'}`}>
              <Plus size={14} className="text-emerald-500" />
            </div>
            <span>New Chat</span>
          </button>
          
          <button 
            onClick={() => setIsOpen(false)}
            aria-label="Collapse sidebar"
            className={`p-2 rounded-lg transition-colors ml-1 ${isDark ? 'hover:bg-zinc-900 text-zinc-500' : 'hover:bg-zinc-200 text-zinc-500'}`}
          >
            <PanelLeft size={18} />
          </button>
        </div>

        {/* Chat List Scroll Container */}
        <div className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
          <div className="px-3 mb-2 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
            Conversations
          </div>
          
          {chats.map((chat) => {
            const isActive = currentChatId === chat.id;
            const isEditing = editingId === chat.id;
            const hasUnread = chat.hasUnreadResponse;
            const isLoading = chat.isLoading;
            const isMenuOpen = openMenuId === chat.id;

            return (
              <div key={chat.id} className="group relative px-2">
                {isEditing ? (
                  <div className={`flex items-center gap-2 px-2 py-1.5 rounded-lg border transition-all ${isDark ? 'bg-zinc-900 border-zinc-700' : 'bg-white border-zinc-300'}`}>
                    <input
                      ref={editInputRef}
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => handleKeyDown(e, chat.id)}
                      onBlur={() => handleSaveRename(chat.id)}
                      className="flex-1 bg-transparent text-sm outline-none w-full"
                    />
                  </div>
                ) : (
                  <div className="relative">
                    <button
                      onClick={() => onSelectChat(chat.id)}
                      className={`flex items-center w-full px-3 py-2 rounded-lg gap-3 text-sm text-left transition-all
                        ${isActive 
                          ? (isDark ? 'bg-zinc-900 text-white' : 'bg-white text-zinc-900 shadow-sm border border-zinc-200') 
                          : (isDark ? 'text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200' : 'text-zinc-600 hover:bg-zinc-200 hover:text-zinc-900')}`}
                    >
                      {isLoading ? (
                        <Loader2 size={14} className="shrink-0 animate-spin text-emerald-500" />
                      ) : (
                        <MessageSquare size={14} className={`shrink-0 ${hasUnread ? 'text-emerald-500' : 'opacity-60'}`} />
                      )}
                      <span className={`truncate flex-1 font-normal ${hasUnread ? 'font-semibold text-emerald-600 dark:text-emerald-400' : ''}`}>
                        {chat.title}
                      </span>
                      
                      <div className="w-6" />
                      
                      {hasUnread && !isActive && !isMenuOpen && (
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-sm" />
                      )}
                    </button>

                    {/* Chat Context Menu Trigger */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuId(isMenuOpen ? null : chat.id);
                      }}
                      aria-label="Chat options"
                      className={`absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-md transition-all z-10
                        ${isMenuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}
                        ${isDark ? 'hover:bg-zinc-800 text-zinc-400' : 'hover:bg-zinc-100 text-zinc-500'}`}
                    >
                      <MoreHorizontal size={14} />
                    </button>

                    {/* Popover Menu for Rename/Delete */}
                    {isMenuOpen && (
                      <>
                        <div 
                          className="fixed inset-0 z-20" 
                          onClick={() => setOpenMenuId(null)} 
                        />
                        <div className={`absolute right-2 top-full mt-1 w-32 rounded-lg border shadow-xl z-30 p-1 animate-in fade-in zoom-in-95 duration-100
                          ${isDark ? 'bg-zinc-900 border-zinc-800' : 'bg-white border-zinc-200'}`}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartRename(chat);
                            }}
                            className={`w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded-md transition-colors
                              ${isDark ? 'hover:bg-zinc-800 text-zinc-300' : 'hover:bg-zinc-100 text-zinc-700'}`}
                          >
                            <Pencil size={12} />
                            Rename
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteChat(chat.id);
                              setOpenMenuId(null);
                            }}
                            className={`w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded-md transition-colors text-red-500
                              ${isDark ? 'hover:bg-zinc-800' : 'hover:bg-red-50'}`}
                          >
                            <Trash2 size={12} />
                            Delete
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Global Action Footer */}
        <div className={`p-4 border-t space-y-2 flex flex-col ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
          <button 
            onClick={toggleTheme}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all
              ${isDark ? 'hover:bg-zinc-900 text-zinc-400' : 'hover:bg-zinc-200 text-zinc-600'}`}
          >
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
            <span>{isDark ? 'Light' : 'Dark'} mode</span>
          </button>

          <button 
            className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all
              ${isDark ? 'hover:bg-zinc-900 text-zinc-400' : 'hover:bg-zinc-200 text-zinc-600'}`}
          >
            <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center text-[10px] text-white">
              {profile.name.charAt(0)}
            </div>
            <span className="truncate">{profile.name}</span>
          </button>
        </div>
      </div>
    </aside>
  );
};

export default React.memo(Sidebar);