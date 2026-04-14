import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const detectImage = async (
  file,
  modelName = 'yolov8n',
  optimization = 'none',
  confThreshold = 0.25,
  iouThreshold = 0.45
) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('optimization', optimization);
  formData.append('conf_threshold', confThreshold);
  formData.append('iou_threshold', iouThreshold);

  const response = await api.post('/detect/image', formData);
  return response.data;
};

export const detectVideo = async (
  file,
  modelName = 'yolov8n',
  optimization = 'none',
  confThreshold = 0.25,
  iouThreshold = 0.45,
  skipFrames = 1
) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('optimization', optimization);
  formData.append('conf_threshold', confThreshold);
  formData.append('iou_threshold', iouThreshold);
  formData.append('skip_frames', skipFrames);

  const response = await api.post('/detect/video', formData);
  return response.data;
};

export const benchmarkModels = async (file, models, optimizations, numRuns = 100) => {
  const formData = new FormData();
  formData.append('test_image', file);
  
  models.forEach(model => {
    formData.append('models', model);
  });
  
  optimizations.forEach(opt => {
    formData.append('optimizations', opt);
  });
  
  formData.append('num_runs', numRuns);

  const response = await api.post('/benchmark', formData);
  return response.data;
};

export const getResultFile = (filename) => {
  return `${API_BASE_URL}/results/${filename}`;
};

export default api;
