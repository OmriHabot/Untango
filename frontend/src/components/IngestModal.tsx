import React, { useState } from 'react';
import { X, Github, FolderOpen, Loader2, CheckCircle, AlertCircle, Terminal } from 'lucide-react';
import { api } from '../api/client';
import { useRepoStore } from '../store/repoStore';

interface Props {
  onClose: () => void;
}

type Tab = 'github' | 'local';

export const IngestModal: React.FC<Props> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<Tab>('github');
  
  // GitHub state
  const [url, setUrl] = useState('');
  const [branch, setBranch] = useState('');
  
  // Local state
  const [localPath, setLocalPath] = useState('');
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    absolute_path: string | null;
    sample_files: string[];
    venv_python: string | null;
    error?: string;
  } | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  
  // Shared state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { pollStatus, fetchRepositories } = useRepoStore();
  const isLocalBackend = api.isLocalBackend();

  const handleValidatePath = async () => {
    if (!localPath) return;
    
    setIsValidating(true);
    setError(null);
    setValidationResult(null);
    
    try {
      const result = await api.validateLocalPath(localPath);
      setValidationResult(result);
      if (!result.valid && result.error) {
        setError(result.error);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to validate path');
    } finally {
      setIsValidating(false);
    }
  };

  const handleSubmitGithub = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.ingestRepository({
        type: 'github',
        location: url,
        branch: branch || undefined
      });
      
      pollStatus(response.repo_id);
      await fetchRepositories();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start ingestion');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitLocal = async () => {
    if (!validationResult?.valid || !validationResult.absolute_path) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.ingestRepository({
        type: 'local',
        location: validationResult.absolute_path
      });
      
      pollStatus(response.repo_id);
      await fetchRepositories();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start ingestion');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-lg shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <h2 className="text-xl font-semibold text-white">Ingest Repository</h2>
          <button 
            onClick={onClose} 
            className="text-slate-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isLoading}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-800">
          <button
            onClick={() => setActiveTab('github')}
            className={`flex-1 px-4 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
              activeTab === 'github'
                ? 'text-purple-400 border-b-2 border-purple-500 bg-purple-500/5'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Github className="w-4 h-4" />
            GitHub
          </button>
          <button
            onClick={() => setActiveTab('local')}
            className={`flex-1 px-4 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
              activeTab === 'local'
                ? 'text-purple-400 border-b-2 border-purple-500 bg-purple-500/5'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <FolderOpen className="w-4 h-4" />
            Local Directory
          </button>
        </div>

        <div className="p-6">
          {/* GitHub Tab */}
          {activeTab === 'github' && (
            <form onSubmit={handleSubmitGithub} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">
                  GitHub URL
                </label>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://github.com/username/repo"
                  className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                  required
                  disabled={isLoading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">
                  Branch (Optional)
                </label>
                <input
                  type="text"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  placeholder="Default (auto-detect)"
                  className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                  disabled={isLoading}
                />
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={onClose}
                  disabled={isLoading}
                  className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-md font-medium flex items-center gap-2 disabled:opacity-50 transition-colors"
                >
                  {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                  {isLoading ? 'Ingesting...' : 'Ingest'}
                </button>
              </div>
            </form>
          )}

          {/* Local Directory Tab */}
          {activeTab === 'local' && (
            <div className="space-y-4">
              <div className="p-4 bg-blue-900/20 border border-blue-800/50 rounded-lg">
                <div className="flex items-start gap-3">
                  <Terminal className="w-5 h-5 text-blue-400 mt-0.5 shrink-0" />
                  <div className="text-sm text-blue-200">
                    <p className="font-medium mb-2">Upload Local Repository via CLI</p>
                    <p className="text-blue-300/80">
                      Use the CLI tool to upload your local repository. It will sync your files and detect virtual environments automatically.
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="bg-slate-800 rounded-lg p-4 font-mono text-sm">
                  <div className="text-slate-500 text-xs mb-2"># One-time upload:</div>
                  <code className="text-green-400 break-all">
                    python scripts/untango_local.py /path/to/project --server {window.location.origin}
                  </code>
                </div>

                <div className="bg-slate-800 rounded-lg p-4 font-mono text-sm">
                  <div className="text-slate-500 text-xs mb-2"># Continuous sync (auto-updates on file changes):</div>
                  <code className="text-green-400 break-all">
                    python scripts/untango_local.py /path/to/project --server {window.location.origin} --watch
                  </code>
                </div>
              </div>

              <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700 text-sm">
                <div className="font-medium text-slate-300 mb-1">Features:</div>
                <ul className="text-slate-400 space-y-1 text-xs">
                  <li>• Auto-detects virtual environments</li>
                  <li>• Respects .gitignore patterns</li>
                  <li>• Watch mode syncs changes in real-time</li>
                </ul>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={onClose}
                  className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          )}

          {/* Error display */}
          {error && (
            <div className="mt-4 text-red-400 text-sm bg-red-400/10 p-3 rounded flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          {isLoading && (
            <div className="mt-4 text-amber-400 text-sm bg-amber-400/10 p-3 rounded flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Ingestion in progress. Please do not refresh the page.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
