import React, { useEffect, useState } from 'react';
import { useRepoStore } from '../store/repoStore';
import { useChatStore } from '../store/chatStore';
import { Plus, Database, Loader2, CheckCircle, XCircle, GitBranch, Presentation } from 'lucide-react';
import clsx from 'clsx';
import { IngestModal } from './IngestModal';

export const Sidebar: React.FC = () => {
  const { repositories, activeRepoId, fetchRepositories, setActiveRepo, checkActiveRepo, ingestionStatuses, viewMode, setViewMode } = useRepoStore();
  const { messages, sendMessage, isLoading: isChatLoading, loadHistory } = useChatStore();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const pollStatus = () => {
    checkActiveRepo();
  };

  useEffect(() => {
    fetchRepositories();
    const interval = setInterval(pollStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load chat history when active repo changes
  useEffect(() => {
    if (activeRepoId && activeRepoId !== 'default') {
      loadHistory();
    }
  }, [activeRepoId]);

  return (
    <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-screen text-slate-300">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h1 className="font-bold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-500" />
          Untango
        </h1>
        <button 
          onClick={() => setIsModalOpen(true)}
          className="p-1.5 hover:bg-slate-800 rounded-md transition-colors"
          title="Add Repository"
        >
          <Plus className="w-5 h-5" />
        </button>
      </div>

      <div className="p-2 border-b border-slate-800">
        <button
          onClick={() => setViewMode(viewMode === 'showcase' ? 'chat' : 'showcase')}
          className={clsx(
            "w-full text-left px-3 py-2 rounded-md text-sm flex items-center gap-3 transition-colors",
            viewMode === 'showcase'
              ? "bg-purple-600 text-white shadow-lg shadow-purple-900/20" 
              : "bg-slate-800/50 text-purple-400 hover:bg-slate-800 hover:text-purple-300"
          )}
        >
          <Presentation className="w-4 h-4" />
          <span className="font-semibold">Showcase</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        <div className="px-2 py-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Repositories
        </div>
        
        {repositories.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-slate-500">
            No repositories yet.
            <br />
            Click + to add one.
          </div>
        )}

        {repositories.map((repo) => {
          const status = ingestionStatuses[repo.repo_id];
          const isActive = activeRepoId === repo.repo_id;
          
          return (
            <button
              key={repo.repo_id}
              onClick={() => {
                setActiveRepo(repo.repo_id);
                setViewMode('chat');
              }}
              className={clsx(
                "w-full text-left px-3 py-2 rounded-md text-sm flex items-center gap-3 transition-colors",
                isActive 
                  ? "bg-purple-500/10 text-purple-400 border border-purple-500/20" 
                  : "hover:bg-slate-800 text-slate-400 hover:text-slate-200"
              )}
            >
              <GitBranch className="w-4 h-4 shrink-0" />
              <div className="flex-1 truncate">
                {repo.name} <span className="text-slate-500 text-xs">({repo.repo_id.substring(0, 6)})</span>
              </div>
              
              {status === 'ingesting' && <Loader2 className="w-3 h-3 animate-spin text-blue-400" />}
              {status === 'completed' && isActive && <CheckCircle className="w-3 h-3 text-green-500" />}
              {status === 'failed' && <XCircle className="w-3 h-3 text-red-500" />}
            </button>
          );
        })}
      </div>

      <div className="p-4 border-t border-slate-800 text-xs text-slate-600">
        v1.0.0 • Local Mode
      </div>

      {isModalOpen && <IngestModal onClose={() => setIsModalOpen(false)} />}
    </div>
  );
};
