import React, { useState } from 'react';
import { X, Github, Loader2 } from 'lucide-react';
import { api } from '../api/client';
import { useRepoStore } from '../store/repoStore';

interface Props {
  onClose: () => void;
}

export const IngestModal: React.FC<Props> = ({ onClose }) => {
  const [url, setUrl] = useState('');
  const [branch, setBranch] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { pollStatus, fetchRepositories } = useRepoStore();

  const handleSubmit = async (e: React.FormEvent) => {
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
      
      // Start polling
      pollStatus(response.repo_id);
      
      // Refresh list (it might show up as pending immediately if backend supports it, 
      // but our list endpoint might not show it until folder exists. 
      // The backend creates the folder immediately though.)
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
      <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-md shadow-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <Github className="w-6 h-6" />
            Ingest Repository
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-1">
              GitHub URL
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://github.com/username/repo"
              className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              required
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
              className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>

          {error && (
            <div className="text-red-400 text-sm bg-red-400/10 p-2 rounded">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-3 mt-6">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-md font-medium flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Ingest
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
