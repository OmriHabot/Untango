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
            <div className="max-w-none">
              <ReactMarkdown
                components={{
                  // Headers with explicit sizing
                  h1: ({ children, ...props }: any) => (
                    <h1 className="text-2xl font-bold text-white mt-6 mb-4 first:mt-0" {...props}>{children}</h1>
                  ),
                  h2: ({ children, ...props }: any) => (
                    <h2 className="text-xl font-semibold text-white mt-5 mb-3 first:mt-0" {...props}>{children}</h2>
                  ),
                  h3: ({ children, ...props }: any) => (
                    <h3 className="text-lg font-semibold text-slate-100 mt-4 mb-2 first:mt-0" {...props}>{children}</h3>
                  ),
                  h4: ({ children, ...props }: any) => (
                    <h4 className="text-base font-semibold text-slate-200 mt-3 mb-2 first:mt-0" {...props}>{children}</h4>
                  ),
                  h5: ({ children, ...props }: any) => (
                    <h5 className="text-sm font-semibold text-slate-200 mt-2 mb-1 first:mt-0" {...props}>{children}</h5>
                  ),
                  h6: ({ children, ...props }: any) => (
                    <h6 className="text-sm font-medium text-slate-300 mt-2 mb-1 first:mt-0" {...props}>{children}</h6>
                  ),
                  // Paragraphs
                  p: ({ children, ...props }: any) => (
                    <p className="text-slate-300 mb-3 last:mb-0 leading-relaxed" {...props}>{children}</p>
                  ),
                  // Lists
                  ul: ({ children, ...props }: any) => (
                    <ul className="list-disc list-inside mb-3 space-y-1 text-slate-300" {...props}>{children}</ul>
                  ),
                  ol: ({ children, ...props }: any) => (
                    <ol className="list-decimal list-inside mb-3 space-y-1 text-slate-300" {...props}>{children}</ol>
                  ),
                  li: ({ children, ...props }: any) => (
                    <li className="text-slate-300" {...props}>{children}</li>
                  ),
                  // Links
                  a: ({ children, href, ...props }: any) => (
                    <a href={href} className="text-purple-400 hover:text-purple-300 underline" target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
                  ),
                  // Blockquotes
                  blockquote: ({ children, ...props }: any) => (
                    <blockquote className="border-l-4 border-purple-500 pl-4 my-3 italic text-slate-400" {...props}>{children}</blockquote>
                  ),
                  // Horizontal rule
                  hr: (props: any) => (
                    <hr className="border-slate-700 my-4" {...props} />
                  ),
                  // Strong/Bold
                  strong: ({ children, ...props }: any) => (
                    <strong className="font-semibold text-white" {...props}>{children}</strong>
                  ),
                  // Emphasis/Italic
                  em: ({ children, ...props }: any) => (
                    <em className="italic text-slate-200" {...props}>{children}</em>
                  ),
                  // Code blocks
                  code({ node, inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <div className="rounded-md overflow-hidden my-3 border border-slate-700">
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
                      <code className="bg-slate-800 px-1.5 py-0.5 rounded text-purple-300 font-mono text-sm" {...props}>
                        {children}
                      </code>
                    );
                  },
                  // Tables
                  table: ({ children, ...props }: any) => (
                    <div className="overflow-x-auto my-3">
                      <table className="min-w-full border border-slate-700 text-sm" {...props}>{children}</table>
                    </div>
                  ),
                  thead: ({ children, ...props }: any) => (
                    <thead className="bg-slate-800" {...props}>{children}</thead>
                  ),
                  th: ({ children, ...props }: any) => (
                    <th className="border border-slate-700 px-3 py-2 text-left font-semibold text-slate-200" {...props}>{children}</th>
                  ),
                  td: ({ children, ...props }: any) => (
                    <td className="border border-slate-700 px-3 py-2 text-slate-300" {...props}>{children}</td>
                  ),
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
