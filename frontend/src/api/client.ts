import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8001/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for LLM calls
});

const healthApi = axios.create({
  baseURL: 'http://127.0.0.1:8001',
  timeout: 5000,
});

export const chatWithSyntera = async (query: string, mode: string = 'auto') => {
  const response = await api.post('/chat', { query, mode });
  return response.data;
};

export const uploadDocument = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/kb/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 minutes for large files
  });
  return response.data;
};

export const getDocuments = async () => {
  const response = await api.get('/kb/documents');
  return response.data;
};

export const getStatus = async () => {
  const response = await api.get('/status');
  return response.data;
};

export const getHealth = async () => {
  try {
    const response = await healthApi.get('/health');
    return response.data;
  } catch {
    return { status: 'OFFLINE', service: 'NexusLLM', components: {} };
  }
};
