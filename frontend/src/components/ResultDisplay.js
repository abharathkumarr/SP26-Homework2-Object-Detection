import React from 'react';
import { CheckCircle, Clock, Zap, Image } from 'lucide-react';
import { getResultFile } from '../services/api';
import './ResultDisplay.css';

const ResultDisplay = ({ result }) => {
  if (!result) return null;

  const { fileType } = result;

  return (
    <div className="result-display">
      <div className="result-header">
        <CheckCircle size={24} />
        <h2>Detection Results</h2>
      </div>

      <div className="result-stats">
        <div className="stat-card">
          <Clock className="stat-icon" />
          <div className="stat-content">
            <span className="stat-label">Inference Time</span>
            <span className="stat-value">
              {fileType === 'video'
                ? `${result.average_inference_time?.toFixed(4)}s`
                : `${result.inference_time?.toFixed(4)}s`}
            </span>
          </div>
        </div>

        <div className="stat-card">
          <Zap className="stat-icon" />
          <div className="stat-content">
            <span className="stat-label">FPS</span>
            <span className="stat-value">
              {fileType === 'video'
                ? result.average_fps?.toFixed(2)
                : result.fps?.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="stat-card">
          <Image className="stat-icon" />
          <div className="stat-content">
            <span className="stat-label">Detections</span>
            <span className="stat-value">
              {fileType === 'video'
                ? Object.values(result.detections_summary || {}).reduce((a, b) => a + b, 0)
                : result.num_detections}
            </span>
          </div>
        </div>
      </div>

      {fileType === 'image' && result.result_image && (
        <div className="result-image-container">
          <h3>Detected Objects</h3>
          <img
            src={getResultFile(result.result_image.split('/').pop())}
            alt="Detection result"
            className="result-image"
          />
        </div>
      )}

      {fileType === 'video' && result.result_video && (
        <div className="result-video-container">
          <h3>Processed Video</h3>
          <video
            src={getResultFile(result.result_video.split('/').pop())}
            controls
            className="result-video"
          >
            Your browser does not support the video tag.
          </video>
        </div>
      )}

      {result.detections && result.detections.length > 0 && (
        <div className="detections-list">
          <h3>Detection Details</h3>
          <div className="detections-grid">
            {result.detections.slice(0, 10).map((det, idx) => (
              <div key={idx} className="detection-item">
                <span className="detection-class">{det.class_name}</span>
                <span className="detection-conf">{(det.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
            {result.detections.length > 10 && (
              <div className="detection-item more">
                +{result.detections.length - 10} more
              </div>
            )}
          </div>
        </div>
      )}

      {result.detections_summary && (
        <div className="detections-summary">
          <h3>Detection Summary</h3>
          <div className="summary-grid">
            {Object.entries(result.detections_summary).map(([className, count]) => (
              <div key={className} className="summary-item">
                <span className="summary-class">{className}</span>
                <span className="summary-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="result-footer">
        <span>Model: <strong>{result.model}</strong></span>
        <span>Optimization: <strong>{result.optimization}</strong></span>
      </div>
    </div>
  );
};

export default ResultDisplay;
