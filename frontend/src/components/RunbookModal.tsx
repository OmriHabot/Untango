import React, { useState, useEffect } from 'react';
import { X, BookOpen, Loader2, Copy, Check, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { api } from '../api/client';
import { useRepoStore } from '../store/repoStore';

interface Props {
  onClose: () => void;
}

export const RunbookModal: React.FC<Props> = ({ onClose }) => {
  const { activeRepoId, repositories } = useRepoStore();
  const [isLoading, setIsLoading] = useState(false);
  const [runbook, setRunbook] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const activeRepo = repositories.find(r => r.repo_id === activeRepoId);

  const loadRunbook = async (forceRegenerate = false) => {
    if (!activeRepo) return;
    setIsLoading(true);
    try {
      if (!forceRegenerate) {
        // Try to get existing runbook first
        const existing = await api.getRunbook(activeRepo.repo_id);
        if (existing && existing.runbook) {
          setRunbook(existing.runbook);
          setIsLoading(false);
          return;
        }
      }
      
      // Generate new
      const response = await api.generateRunbook(activeRepo.path, activeRepo.name, activeRepo.repo_id);
      setRunbook(response.runbook);
    } catch (error) {
      console.error('Failed to generate runbook:', error);
      setRunbook('# Error\nFailed to generate runbook. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (runbook) {
      navigator.clipboard.writeText(runbook);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  useEffect(() => {
    loadRunbook();
  }, []);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-4xl h-[80vh] shadow-xl flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-slate-800">
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-purple-400" />
            Runbook Generator
          </h2>
          <div className="flex items-center gap-2">
            {runbook && (
              <>
                <button 
                  onClick={() => loadRunbook(true)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
                  title="Regenerate Runbook"
                  disabled={isLoading}
                >
                  <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
                <button 
                  onClick={handleCopy}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
                  title="Copy Markdown"
                >
                  {copied ? <Check className="w-5 h-5 text-green-500" /> : <Copy className="w-5 h-5" />}
                </button>
              </>
            )}
            <button onClick={onClose} className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 bg-slate-950">
          {isLoading ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
              <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
              <p>Analyzing repository and generating runbook...</p>
              <p className="text-xs">This may take a minute.</p>
            </div>
          ) : runbook ? (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <div className="rounded-md overflow-hidden my-2 border border-slate-700">
                        <div className="bg-slate-900 px-4 py-1 text-xs text-slate-400 border-b border-slate-700 flex justify-between items-center">
                          <span>{match[1]}</span>
                        </div>
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          customStyle={{ margin: 0, borderRadius: 0, background: '#0f172a' }}
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      </div>
                    ) : (
                      <code className="bg-slate-800 px-1 py-0.5 rounded text-purple-300 font-mono text-sm" {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {runbook}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="text-center text-slate-500">
              Failed to load.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
