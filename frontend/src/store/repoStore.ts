import { create } from 'zustand';
import { api, Repository } from '../api/client';

// Watch state for a single repository
export interface WatchState {
  isWatching: boolean;
  lastSyncTime: Date | null;
  syncCount: number;
}

// Module-level storage (not in zustand state - can't serialize FileSystemDirectoryHandle)
const directoryHandles: Map<string, FileSystemDirectoryHandle> = new Map();
const watchIntervals: Map<string, number> = new Map();
const fileHashes: Map<string, Record<string, string>> = new Map();
const repoIdToName: Map<string, string> = new Map();

interface RepoState {
  repositories: Repository[];
  activeRepoId: string | null;
  isLoading: boolean;
  ingestionStatuses: Record<string, string>;
  watchStates: Record<string, WatchState>;
  
  fetchRepositories: () => Promise<void>;
  setActiveRepo: (repoId: string) => Promise<void>;
  checkActiveRepo: () => Promise<void>;
  pollStatus: (repoId: string) => Promise<void>;
  
  // Watch management
  startWatch: (repoId: string, directoryHandle: FileSystemDirectoryHandle, initialHashes: Record<string, string>) => void;
  stopWatch: (repoId: string) => void;
  updateWatchState: (repoId: string, updates: Partial<WatchState>) => void;
  getDirectoryHandle: (repoId: string) => FileSystemDirectoryHandle | null;
  
  viewMode: 'chat' | 'showcase';
  setViewMode: (mode: 'chat' | 'showcase') => void;

  activeShowcaseTab: 'docs' | 'agent' | 'qualitative' | 'hybrid' | 'evaluation' | 'chunking';
  setActiveShowcaseTab: (tab: 'docs' | 'agent' | 'qualitative' | 'hybrid' | 'evaluation' | 'chunking') => void;
}

export const useRepoStore = create<RepoState>((set, get) => ({
  repositories: [],
  activeRepoId: null,
  isLoading: false,
  ingestionStatuses: {},
  watchStates: {},
  viewMode: 'chat',
  activeShowcaseTab: 'docs',

  setActiveShowcaseTab: (tab) => set({ activeShowcaseTab: tab }),
  setViewMode: (mode) => set({ viewMode: mode }),

  fetchRepositories: async () => {
    set({ isLoading: true });
    try {
      const data = await api.listRepositories();
      set({ repositories: data.repositories });
      // Update repoIdToName map
      data.repositories.forEach(repo => {
        repoIdToName.set(repo.repo_id, repo.name);
      });
    } catch (error) {
      console.error('Failed to fetch repositories:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  setActiveRepo: async (repoId: string) => {
    try {
      await api.setActiveRepository(repoId);
      set({ activeRepoId: repoId });
    } catch (error) {
      console.error('Failed to set active repo:', error);
    }
  },

  checkActiveRepo: async () => {
    try {
      const data = await api.getActiveRepository();
      if (data.active_repo_id && data.active_repo_id !== 'default') {
        set({ activeRepoId: data.active_repo_id });
      }
    } catch (error) {
      console.error('Failed to check active repo:', error);
    }
  },

  pollStatus: async (repoId: string) => {
    // Simple polling implementation
    // In a real app, we'd use a more robust poller or websockets
    const check = async () => {
      try {
        const data = await api.getRepoStatus(repoId);
        set((state) => ({
          ingestionStatuses: { ...state.ingestionStatuses, [repoId]: data.status }
        }));
        
        if (data.status === 'pending' || data.status === 'ingesting') {
          setTimeout(check, 2000);
        } else if (data.status === 'completed') {
          // Refresh list to ensure path is correct if needed
          get().fetchRepositories();
          // Auto-select if it's the only one or user waiting?
          // For now just update status.
        }
      } catch (error) {
        console.error('Polling failed:', error);
      }
    };
    check();
  },

  // Watch management functions
  startWatch: (repoId, directoryHandle, initialHashes) => {
    // Store the handle and hashes
    directoryHandles.set(repoId, directoryHandle);
    fileHashes.set(repoId, initialHashes);
    
    // Initialize watch state
    set((state) => ({
      watchStates: {
        ...state.watchStates,
        [repoId]: {
          isWatching: true,
          lastSyncTime: null,
          syncCount: 0
        }
      }
    }));
    
    // Import utilities dynamically to avoid circular dependencies
    import('../utils/localDirectoryUtils').then(async ({ scanDirectoryWithSummary, computeFileHashes, findChangedFiles, createZipBundle }) => {
      // Start polling interval
      const intervalId = window.setInterval(async () => {
        const handle = directoryHandles.get(repoId);
        const currentHashes = fileHashes.get(repoId);
        
        if (!handle || !currentHashes) {
          get().stopWatch(repoId);
          return;
        }
        
        try {
          // Re-scan directory
          const result = await scanDirectoryWithSummary(handle);
          const newHashes = await computeFileHashes(result.files);
          
          // Find changed files
          const changedPaths = findChangedFiles(currentHashes, newHashes);
          
          if (changedPaths.length > 0) {
            // Get the changed files
            const changedFiles = result.files.filter(f => changedPaths.includes(f.path));
            
            // Create bundle of changed files
            const bundle = await createZipBundle(changedFiles);
            
            // Sync to server
            await api.syncRepository(bundle, repoId);
            
            // Update state
            fileHashes.set(repoId, newHashes);
            set((state) => ({
              watchStates: {
                ...state.watchStates,
                [repoId]: {
                  ...state.watchStates[repoId],
                  lastSyncTime: new Date(),
                  syncCount: (state.watchStates[repoId]?.syncCount || 0) + 1
                }
              }
            }));
          }
        } catch (err) {
          console.error('Watch sync error:', err);
        }
      }, 3000);
      
      watchIntervals.set(repoId, intervalId);
    });
  },

  stopWatch: (repoId) => {
    // Clear interval
    const intervalId = watchIntervals.get(repoId);
    if (intervalId) {
      clearInterval(intervalId);
      watchIntervals.delete(repoId);
    }
    
    // Clear stored data
    directoryHandles.delete(repoId);
    fileHashes.delete(repoId);
    
    // Update state
    set((state) => ({
      watchStates: {
        ...state.watchStates,
        [repoId]: {
          ...state.watchStates[repoId],
          isWatching: false
        }
      }
    }));
  },

  updateWatchState: (repoId, updates) => {
    set((state) => ({
      watchStates: {
        ...state.watchStates,
        [repoId]: {
          ...state.watchStates[repoId],
          ...updates
        }
      }
    }));
  },

  getDirectoryHandle: (repoId) => {
    return directoryHandles.get(repoId) || null;
  }
}));
