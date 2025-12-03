import React, { useEffect, useRef, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { useRepoStore } from '../store/repoStore';
import { MessageBubble } from './MessageBubble';
import { Send, Loader2, Eraser, BookOpen } from 'lucide-react';
import { RunbookModal } from './RunbookModal';

export const ChatArea: React.FC = () => {
  const { messages, sendMessage, isStreaming, clearMessages } = useChatStore();
  const { activeRepoId } = useRepoStore();
  const [input, setInput] = useState('');
  const [isRunbookOpen, setIsRunbookOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    
    const content = input;
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    
    await sendMessage(content);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="h-14 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Context:</span>
          <span className="font-mono text-purple-400 bg-purple-400/10 px-2 py-0.5 rounded text-sm">
            {activeRepoId?.substring(0, 8)}...
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setIsRunbookOpen(true)}
            className="flex items-center gap-2 text-slate-400 hover:text-purple-400 transition-colors px-3 py-1.5 rounded-md hover:bg-slate-900 text-sm font-medium"
            title="Generate Runbook"
          >
            <BookOpen className="w-4 h-4" />
            <span className="hidden sm:inline">Runbook</span>
          </button>
          <button 
            onClick={clearMessages}
            className="text-slate-500 hover:text-red-400 transition-colors p-2 rounded-md hover:bg-slate-900"
            title="Clear Chat"
          >
            <Eraser className="w-4 h-4" />
          </button>
        </div>
      </div>

      {isRunbookOpen && <RunbookModal onClose={() => setIsRunbookOpen(false)} />}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
            <div className="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center border border-slate-800">
              <Send className="w-6 h-6 text-slate-600" />
            </div>
            <p>Ask anything about the repository...</p>
            <div className="flex gap-2">
              <button 
                onClick={() => setInput("What is the structure of this repository?")}
                className="text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full hover:border-purple-500/50 hover:text-purple-300 transition-colors"
              >
                Repo Structure
              </button>
              <button 
                onClick={() => setInput("Explain the main functionality.")}
                className="text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full hover:border-purple-500/50 hover:text-purple-300 transition-colors"
              >
                Main Functionality
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-slate-800 bg-slate-950">
        <div className="max-w-4xl mx-auto relative">
          {isStreaming && (
            <div className="absolute -top-10 left-0 right-0 flex justify-center">
              <div className="bg-purple-900/50 text-purple-200 text-xs px-3 py-1.5 rounded-full flex items-center gap-2 border border-purple-500/30 backdrop-blur-sm animate-pulse">
                <Loader2 className="w-3 h-3 animate-spin" />
                Generating response...
              </div>
            </div>
          )}
          
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? "Waiting for response..." : "Ask a question..."}
            rows={1}
            className="w-full bg-slate-900 border border-slate-700 rounded-xl pl-4 pr-12 py-3 text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 resize-none max-h-48 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-900/50"
            disabled={isStreaming}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="absolute right-2 bottom-2 p-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg disabled:opacity-50 disabled:bg-slate-800 disabled:text-slate-500 transition-all"
          >
            {isStreaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <div className="text-center mt-2 text-xs text-slate-600">
          AI can make mistakes. Review generated code.
        </div>
      </div>
    </div>
  );
};
