import React, { useState } from 'react';
import { X, Send, CheckCircle } from 'lucide-react';
import { Theme } from '../types';

const API_URL = import.meta.env.VITE_BACKEND_URL !== undefined
  ? import.meta.env.VITE_BACKEND_URL
  : (import.meta.env.PROD ? '' : 'http://localhost:9481');

interface ContactModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: Theme;
}

type ContactType = 'feedback' | 'contact';
type Status = 'idle' | 'sending' | 'success' | 'error';

const ContactModal: React.FC<ContactModalProps> = ({ isOpen, onClose, theme }) => {
  const isDark = theme === 'dark';

  const [contactType, setContactType] = useState<ContactType>('feedback');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState<Status>('idle');

  const handleClose = () => {
    if (status === 'sending') return;
    onClose();
    // Only reset after a successful send — preserve drafts on accidental close
    if (status === 'success') {
      setTimeout(() => {
        setStatus('idle');
        setName('');
        setEmail('');
        setMessage('');
        setContactType('feedback');
      }, 300);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !message.trim()) return;

    setStatus('sending');
    try {
      const res = await fetch(`${API_URL}/api/contact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contact_type: contactType,
          name: name.trim(),
          email: email.trim(),
          message: message.trim(),
        }),
      });
      if (!res.ok) throw new Error('Request failed');
      setStatus('success');
    } catch {
      setStatus('error');
    }
  };

  if (!isOpen) return null;

  const inputBase = `w-full px-3 py-2 rounded-lg text-sm outline-none transition-colors border
    ${isDark
      ? 'bg-zinc-800 border-zinc-700 text-zinc-100 placeholder-zinc-500 focus:border-emerald-500'
      : 'bg-zinc-50 border-zinc-200 text-zinc-900 placeholder-zinc-400 focus:border-emerald-500'}`;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300"
      aria-modal="true"
      role="dialog"
      aria-labelledby="contact-title"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Card */}
      <div
        className={`relative z-10 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden
          ${isDark ? 'bg-zinc-900 text-zinc-200' : 'bg-white text-zinc-900'}
          animate-in zoom-in-95 fade-in duration-300`}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 pt-6 pb-4">
          <div>
            <h2
              id="contact-title"
              className={`text-base font-semibold tracking-tight ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}
            >
              {contactType === 'feedback' ? 'Leave Feedback' : 'Get in Touch'}
            </h2>
            <p className={`mt-0.5 text-xs ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
              {contactType === 'feedback'
                ? 'Help improve this assistant'
                : 'Send a message to Sergio'}
            </p>
          </div>
          <button
            onClick={handleClose}
            aria-label="Close"
            className={`ml-4 mt-0.5 shrink-0 p-1.5 rounded-lg transition-colors
              ${isDark ? 'hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300' : 'hover:bg-zinc-100 text-zinc-400 hover:text-zinc-600'}`}
          >
            <X size={16} />
          </button>
        </div>

        {status === 'success' ? (
          <div className="px-6 pb-8 flex flex-col items-center gap-3 text-center">
            <CheckCircle size={40} className="text-emerald-500" />
            <p className={`font-medium ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}>Message sent!</p>
            <p className={`text-sm ${isDark ? 'text-zinc-400' : 'text-zinc-500'}`}>
              Thanks for reaching out. I'll get back to you soon.
            </p>
            <button
              onClick={handleClose}
              className="mt-2 px-5 py-2 rounded-lg text-sm font-medium bg-emerald-500 hover:bg-emerald-600 text-white transition-colors"
            >
              Close
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="px-6 pb-6 space-y-4">
            {/* Type toggle */}
            <div className={`flex rounded-lg p-1 gap-1 ${isDark ? 'bg-zinc-800' : 'bg-zinc-100'}`}>
              {(['feedback', 'contact'] as ContactType[]).map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => setContactType(type)}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all
                    ${contactType === type
                      ? (isDark ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'bg-white text-zinc-900 shadow-sm')
                      : (isDark ? 'text-zinc-400 hover:text-zinc-300' : 'text-zinc-500 hover:text-zinc-700')}`}
                >
                  {type === 'feedback' ? 'Feedback' : 'Get in Touch'}
                </button>
              ))}
            </div>

            <div className="space-y-3">
              <input
                type="text"
                placeholder="Your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className={inputBase}
              />
              <input
                type="email"
                placeholder="Your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className={inputBase}
              />
              <div className={`rounded-lg border overflow-hidden transition-colors
                ${isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-zinc-50 border-zinc-200'}`}>
                <textarea
                  placeholder={contactType === 'feedback'
                    ? 'What could be improved? What did you like?'
                    : 'What would you like to discuss?'}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  required
                  rows={4}
                  className={`w-full px-3 py-2 text-sm outline-none resize-none overflow-y-auto bg-transparent
                    ${isDark ? 'text-zinc-100 placeholder-zinc-500' : 'text-zinc-900 placeholder-zinc-400'}`}
                />
              </div>
            </div>

            {status === 'error' && (
              <p className="text-xs text-red-500">
                Something went wrong. Please try again.
              </p>
            )}

            <button
              type="submit"
              disabled={status === 'sending' || !name.trim() || !email.trim() || !message.trim()}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium
                bg-emerald-500 hover:bg-emerald-600 text-white transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {status === 'sending' ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Sending...
                </>
              ) : (
                <>
                  <Send size={14} />
                  Send Message
                </>
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default ContactModal;
