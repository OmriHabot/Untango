import axios from 'axios';

const API_URL = 'http://localhost:8001';

export const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Repository {
  repo_id: string;
  name: string;
  path: string;
  source_location?: string;
  source_type?: string;
}

export interface ListRepositoriesResponse {
  repositories: Repository[];
  count: number;
}

export interface RepoSource {
  type: 'github' | 'local';
  location: string;
  branch?: string;
}

export interface IngestResponse {
  repo_id: string;
  repo_name: string;
  status: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

export const api = {
  listRepositories: async () => {
    const response = await client.get<ListRepositoriesResponse>('/list-repositories');
    return response.data;
  },

  ingestRepository: async (source: RepoSource) => {
    const response = await client.post<IngestResponse>('/ingest-repository', {
      source,
      parse_dependencies: true,
    });
    return response.data;
  },

  setActiveRepository: async (repoId: string) => {
    const response = await client.post('/set-active-repository', { repo_id: repoId });
    return response.data;
  },

  getActiveRepository: async () => {
    const response = await client.get<{ active_repo_id: string }>('/active-repository');
    return response.data;
  },

  getRepoStatus: async (repoId: string) => {
    const response = await client.get<{ status: string }>(`/repository/${repoId}/status`);
    return response.data;
  },

  generateRunbook: async (repoPath: string, repoName: string, repoId: string) => {
    const response = await client.post('/generate-runbook', {
      repo_path: repoPath,
      repo_name: repoName,
      repo_id: repoId
    });
    return response.data;
  },

  getRunbook: async (repoId: string) => {
    try {
      const response = await client.get<{ runbook: string }>(`/runbook/${repoId}`);
      return response.data;
    } catch (error) {
      return null;
    }
  },

  getChatHistory: async () => {
    const response = await client.get(`${API_URL}/chat/history`);
    return response.data;
  },

  clearChatHistory: async () => {
    const response = await client.delete(`${API_URL}/chat/history`);
    return response.data;
  },
  
  // Showcase / Debug API
  debugChunkCode: async (code: string) => {
    const response = await client.post('/api/debug/chunk', { code });
    return response.data;
  },
  
  debugSearchExplain: async (query: string) => {
    const response = await client.post('/api/debug/search-explain', { query, n_results: 5 });
    return response.data;
  },

  getReadme: async () => {
    const response = await client.get<{ content: string }>('/api/docs/readme');
    return response.data;
  },

  validateLocalPath: async (path: string) => {
    const response = await client.post<{
      valid: boolean;
      absolute_path: string | null;
      sample_files: string[];
      venv_python: string | null;
      error?: string;
    }>('/api/validate-local-path', { path });
    return response.data;
  },

  isLocalBackend: () => {
    return API_URL.includes('localhost') || API_URL.includes('127.0.0.1');
  },

  // Local directory upload via File System Access API
  uploadLocalDirectory: async (bundle: Blob, repoName: string, sourcePath: string) => {
    const formData = new FormData();
    formData.append('bundle', bundle, 'repo.zip');
    formData.append('repo_name', repoName);
    formData.append('source_path', sourcePath);
    formData.append('venv_python', '');

    const response = await client.post<IngestResponse>('/api/ingest-local-upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  // Sync changed files for watch mode
  syncRepository: async (bundle: Blob, repoId: string) => {
    const formData = new FormData();
    formData.append('bundle', bundle, 'changes.zip');
    formData.append('repo_id', repoId);

    const response = await client.post<{ status: string; synced_files: number }>(
      '/api/sync-repository', 
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  }
};
