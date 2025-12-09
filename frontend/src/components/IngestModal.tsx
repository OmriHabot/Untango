import React, { useState, useRef, useCallback, useEffect } from 'react';
import { X, Github, FolderOpen, Loader2, AlertCircle, Terminal, Eye, EyeOff, RefreshCw, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { useRepoStore } from '../store/repoStore';
import {
  isFileSystemAccessSupported,
  pickDirectory,
  scanDirectoryWithSummary,
  createZipBundle,
  computeFileHashes,
  findChangedFiles,
  formatFileSize,
  ScannedFile
} from '../utils/localDirectoryUtils';

interface Props {
  onClose: () => void;
}

type Tab = 'github' | 'local';

export const IngestModal: React.FC<Props> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<Tab>('github');
  
  // GitHub state
  const [url, setUrl] = useState('');
  const [branch, setBranch] = useState('');
  
  // Local state - directory picker
  const [directoryHandle, setDirectoryHandle] = useState<FileSystemDirectoryHandle | null>(null);
  const [scannedFiles, setScannedFiles] = useState<ScannedFile[]>([]);
  const [directoryName, setDirectoryName] = useState<string>('');
  const [totalSize, setTotalSize] = useState<number>(0);
  const [isScanning, setIsScanning] = useState(false);
  
  // Watch mode state
  const [watchEnabled, setWatchEnabled] = useState(false);
  const [isWatching, setIsWatching] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);
  const [syncCount, setSyncCount] = useState(0);
  const fileHashesRef = useRef<Record<string, string>>({});
  const watchIntervalRef = useRef<number | null>(null);
  const repoIdRef = useRef<string | null>(null);
  
  // Shared state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  
  const { pollStatus, fetchRepositories } = useRepoStore();
  const fsApiSupported = isFileSystemAccessSupported();

  // Cleanup watch interval on unmount
  useEffect(() => {
    return () => {
      if (watchIntervalRef.current) {
        clearInterval(watchIntervalRef.current);
      }
    };
  }, []);

  const handlePickDirectory = async () => {
    setError(null);
    setSuccessMessage(null);
    
    try {
      const handle = await pickDirectory();
      if (!handle) return; // User cancelled
      
      setDirectoryHandle(handle);
      setIsScanning(true);
      
      const result = await scanDirectoryWithSummary(handle);
      setScannedFiles(result.files);
      setDirectoryName(result.directoryName);
      setTotalSize(result.totalSize);
      
      // Compute initial hashes for watch mode
      const hashes = await computeFileHashes(result.files);
      fileHashesRef.current = hashes;
      
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to access directory';
      setError(message);
    } finally {
      setIsScanning(false);
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
    } catch (err: unknown) {
      const message = err instanceof Error 
        ? err.message 
        : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to start ingestion';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitLocal = async () => {
    if (scannedFiles.length === 0 || !directoryHandle) return;

    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      // Create zip bundle
      const bundle = await createZipBundle(scannedFiles);
      
      // Upload to server
      const response = await api.uploadLocalDirectory(
        bundle,
        directoryName,
        `local://${directoryName}` // Source path identifier
      );
      
      repoIdRef.current = response.repo_id;
      pollStatus(response.repo_id);
      await fetchRepositories();
      
      if (watchEnabled) {
        // Start watch mode instead of closing
        startWatching();
        setSuccessMessage(`Repository "${directoryName}" ingested. Watch mode active.`);
      } else {
        onClose();
      }
    } catch (err: unknown) {
      const message = err instanceof Error 
        ? err.message 
        : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to upload directory';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const startWatching = useCallback(() => {
    if (!directoryHandle || watchIntervalRef.current) return;
    
    setIsWatching(true);
    
    // Poll every 3 seconds
    watchIntervalRef.current = window.setInterval(async () => {
      try {
        // Re-scan directory
        const result = await scanDirectoryWithSummary(directoryHandle);
        const newHashes = await computeFileHashes(result.files);
        
        // Find changed files
        const changedPaths = findChangedFiles(fileHashesRef.current, newHashes);
        
        if (changedPaths.length > 0 && repoIdRef.current) {
          // Get the changed files
          const changedFiles = result.files.filter(f => changedPaths.includes(f.path));
          
          // Create bundle of changed files
          const bundle = await createZipBundle(changedFiles);
          
          // Sync to server
          await api.syncRepository(bundle, repoIdRef.current);
          
          // Update state
          fileHashesRef.current = newHashes;
          setScannedFiles(result.files);
          setLastSyncTime(new Date());
          setSyncCount(prev => prev + 1);
        }
      } catch (err) {
        console.error('Watch sync error:', err);
      }
    }, 3000);
  }, [directoryHandle]);

  const stopWatching = useCallback(() => {
    if (watchIntervalRef.current) {
      clearInterval(watchIntervalRef.current);
      watchIntervalRef.current = null;
    }
    setIsWatching(false);
  }, []);

  const handleClearDirectory = () => {
    stopWatching();
    setDirectoryHandle(null);
    setScannedFiles([]);
    setDirectoryName('');
    setTotalSize(0);
    setSuccessMessage(null);
    fileHashesRef.current = {};
    repoIdRef.current = null;
    setSyncCount(0);
    setLastSyncTime(null);
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-lg shadow-xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800 shrink-0">
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
        <div className="flex border-b border-slate-800 shrink-0">
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

        <div className="p-6 overflow-y-auto flex-1">
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
              {/* Show directory picker if supported, otherwise show CLI instructions */}
              {fsApiSupported ? (
                <>
                  {/* Directory Picker Section */}
                  {!directoryHandle ? (
                    <div className="text-center py-8">
                      <FolderOpen className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                      <p className="text-slate-400 mb-4">
                        Select a local directory to ingest
                      </p>
                      <button
                        onClick={handlePickDirectory}
                        disabled={isScanning}
                        className="px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium flex items-center gap-2 mx-auto disabled:opacity-50 transition-colors"
                      >
                        {isScanning ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Scanning...
                          </>
                        ) : (
                          <>
                            <FolderOpen className="w-5 h-5" />
                            Choose Directory
                          </>
                        )}
                      </button>
                    </div>
                  ) : (
                    <>
                      {/* Selected Directory Info */}
                      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <FolderOpen className="w-5 h-5 text-purple-400" />
                            <span className="font-medium text-white">{directoryName}</span>
                          </div>
                          <button
                            onClick={handleClearDirectory}
                            disabled={isLoading}
                            className="text-slate-400 hover:text-red-400 text-sm disabled:opacity-50"
                          >
                            Clear
                          </button>
                        </div>
                        <div className="text-sm text-slate-400 space-y-1">
                          <p>{scannedFiles.length} files • {formatFileSize(totalSize)}</p>
                        </div>
                        
                        {/* File preview */}
                        <div className="mt-3 max-h-32 overflow-y-auto">
                          <div className="text-xs text-slate-500 space-y-0.5">
                            {scannedFiles.slice(0, 10).map((f, i) => (
                              <div key={i} className="truncate">{f.path}</div>
                            ))}
                            {scannedFiles.length > 10 && (
                              <div className="text-slate-600">
                                ... and {scannedFiles.length - 10} more files
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Watch Mode Toggle */}
                      <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                        <div className="flex items-center gap-3">
                          {watchEnabled ? (
                            <Eye className="w-5 h-5 text-green-400" />
                          ) : (
                            <EyeOff className="w-5 h-5 text-slate-500" />
                          )}
                          <div>
                            <p className="text-sm font-medium text-white">Watch for changes</p>
                            <p className="text-xs text-slate-400">Auto-sync when files change</p>
                          </div>
                        </div>
                        <button
                          onClick={() => setWatchEnabled(!watchEnabled)}
                          disabled={isLoading || isWatching}
                          className={`relative w-12 h-6 rounded-full transition-colors ${
                            watchEnabled ? 'bg-green-600' : 'bg-slate-600'
                          } disabled:opacity-50`}
                        >
                          <div
                            className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                              watchEnabled ? 'translate-x-7' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Watch Status */}
                      {isWatching && (
                        <div className="flex items-center justify-between p-3 bg-green-900/20 rounded-lg border border-green-800/50">
                          <div className="flex items-center gap-2">
                            <RefreshCw className="w-4 h-4 text-green-400 animate-spin" />
                            <div>
                              <span className="text-sm text-green-300">Watching for changes</span>
                              {lastSyncTime && (
                                <p className="text-xs text-green-400/70">
                                  Last sync: {lastSyncTime.toLocaleTimeString()} ({syncCount} syncs)
                                </p>
                              )}
                            </div>
                          </div>
                          <button
                            onClick={stopWatching}
                            className="px-3 py-1 text-sm bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded transition-colors"
                          >
                            Stop
                          </button>
                        </div>
                      )}

                      {/* Submit Buttons */}
                      <div className="flex justify-end gap-3 pt-2">
                        <button
                          onClick={onClose}
                          disabled={isLoading}
                          className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors disabled:opacity-50"
                        >
                          {isWatching ? 'Close (Keep Watching)' : 'Cancel'}
                        </button>
                        {!isWatching && (
                          <button
                            onClick={handleSubmitLocal}
                            disabled={isLoading || scannedFiles.length === 0}
                            className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-md font-medium flex items-center gap-2 disabled:opacity-50 transition-colors"
                          >
                            {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                            {isLoading ? 'Uploading...' : watchEnabled ? 'Ingest & Watch' : 'Ingest'}
                          </button>
                        )}
                      </div>
                    </>
                  )}
                </>
              ) : (
                /* Fallback: CLI Instructions for unsupported browsers */
                <>
                  <div className="p-4 bg-amber-900/20 border border-amber-800/50 rounded-lg mb-4">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
                      <div className="text-sm text-amber-200">
                        <p className="font-medium mb-1">Browser Not Supported</p>
                        <p className="text-amber-300/80">
                          Your browser doesn't support the File System Access API. 
                          Please use Chrome, Edge, or Opera, or use the CLI tool below.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-blue-900/20 border border-blue-800/50 rounded-lg">
                    <div className="flex items-start gap-3">
                      <Terminal className="w-5 h-5 text-blue-400 mt-0.5 shrink-0" />
                      <div className="text-sm text-blue-200">
                        <p className="font-medium mb-2">Upload Local Repository via CLI</p>
                        <p className="text-blue-300/80">
                          Use the CLI tool to upload your local repository.
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

                  <div className="flex justify-end pt-2">
                    <button
                      onClick={onClose}
                      className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
                    >
                      Close
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Success message */}
          {successMessage && (
            <div className="mt-4 text-green-400 text-sm bg-green-400/10 p-3 rounded flex items-start gap-2">
              <CheckCircle className="w-4 h-4 mt-0.5 shrink-0" />
              {successMessage}
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
