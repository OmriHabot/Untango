import { create } from 'zustand';
import { api, Repository } from '../api/client';

interface RepoState {
  repositories: Repository[];
  activeRepoId: string | null;
  isLoading: boolean;
  ingestionStatuses: Record<string, string>;
  
  fetchRepositories: () => Promise<void>;
  setActiveRepo: (repoId: string) => Promise<void>;
  checkActiveRepo: () => Promise<void>;
  pollStatus: (repoId: string) => Promise<void>;
}

export const useRepoStore = create<RepoState>((set, get) => ({
  repositories: [],
  activeRepoId: null,
  isLoading: false,
  ingestionStatuses: {},

  fetchRepositories: async () => {
    set({ isLoading: true });
    try {
      const data = await api.listRepositories();
      set({ repositories: data.repositories });
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
  }
}));
