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

  generateRunbook: async (repoPath: string, repoName: string) => {
    const response = await client.post('/generate-runbook', {
      repo_path: repoPath,
      repo_name: repoName
    });
    return response.data;
  },
};
