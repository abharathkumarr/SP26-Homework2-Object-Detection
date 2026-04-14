import React from 'react';
import { Settings } from 'lucide-react';
import './ModelSelector.css';

const ModelSelector = ({
  selectedModel,
  setSelectedModel,
  selectedOptimization,
  setSelectedOptimization,
  confThreshold,
  setConfThreshold,
  iouThreshold,
  setIouThreshold
}) => {
  const models = [
    { value: 'yolov8n', label: 'YOLOv8 Nano (Fastest)' },
    { value: 'yolov8s', label: 'YOLOv8 Small' },
    { value: 'yolov8m', label: 'YOLOv8 Medium' },
    { value: 'yolov8l', label: 'YOLOv8 Large' },
    { value: 'yolov11n', label: 'YOLOv11 Nano (Latest)' },
    { value: 'yolov11s', label: 'YOLOv11 Small' },
  ];

  const optimizations = [
    { value: 'none', label: 'No Optimization (PyTorch)' },
    { value: 'onnx', label: 'ONNX Runtime' },
    { value: 'tensorrt', label: 'TensorRT (GPU Only)' },
    { value: 'torchscript', label: 'TorchScript' },
  ];

  return (
    <div className="model-selector">
      <div className="selector-header">
        <Settings size={24} />
        <h2>Model Configuration</h2>
      </div>

      <div className="selector-grid">
        <div className="selector-group">
          <label>Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="selector-input"
          >
            {models.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
        </div>

        <div className="selector-group">
          <label>Optimization</label>
          <select
            value={selectedOptimization}
            onChange={(e) => setSelectedOptimization(e.target.value)}
            className="selector-input"
          >
            {optimizations.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="selector-group">
          <label>Confidence Threshold: {confThreshold.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            className="slider"
          />
        </div>

        <div className="selector-group">
          <label>IoU Threshold: {iouThreshold.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
            className="slider"
          />
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;
