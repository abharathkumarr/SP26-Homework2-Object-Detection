import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader } from 'lucide-react';
import FileUpload from './FileUpload';
import './BenchmarkView.css';

const BenchmarkView = ({ onBenchmark, loading, result, error }) => {
  const [selectedModels, setSelectedModels] = useState(['yolov8n', 'yolov8s']);
  const [selectedOptimizations, setSelectedOptimizations] = useState(['none', 'onnx']);
  const [uploadedFile, setUploadedFile] = useState(null);

  const models = [
    'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l',
    'yolov11n', 'yolov11s', 'yolov11m'
  ];

  const optimizations = ['none', 'onnx', 'tensorrt', 'torchscript'];

  const handleModelToggle = (model) => {
    setSelectedModels(prev =>
      prev.includes(model)
        ? prev.filter(m => m !== model)
        : [...prev, model]
    );
  };

  const handleOptimizationToggle = (opt) => {
    setSelectedOptimizations(prev =>
      prev.includes(opt)
        ? prev.filter(o => o !== opt)
        : [...prev, opt]
    );
  };

  const handleFileUpload = (file) => {
    setUploadedFile(file);
  };

  const handleRunBenchmark = () => {
    if (uploadedFile && selectedModels.length > 0 && selectedOptimizations.length > 0) {
      onBenchmark(uploadedFile, selectedModels, selectedOptimizations);
    }
  };

  const prepareChartData = () => {
    if (!result || !result.benchmark_results) return [];

    return result.benchmark_results
      .filter(r => !r.error)
      .map(r => ({
        name: `${r.model} (${r.optimization})`,
        FPS: parseFloat(r.fps.toFixed(2)),
        'Inference Time (ms)': parseFloat((r.avg_inference_time * 1000).toFixed(2))
      }));
  };

  return (
    <div className="benchmark-view">
      <h2>Model Benchmark</h2>
      <p className="benchmark-description">
        Compare performance of different models and optimization backends
      </p>

      <div className="benchmark-section">
        <h3>Select Models</h3>
        <div className="option-grid">
          {models.map(model => (
            <button
              key={model}
              className={`option-button ${selectedModels.includes(model) ? 'selected' : ''}`}
              onClick={() => handleModelToggle(model)}
            >
              {model}
            </button>
          ))}
        </div>
      </div>

      <div className="benchmark-section">
        <h3>Select Optimizations</h3>
        <div className="option-grid">
          {optimizations.map(opt => (
            <button
              key={opt}
              className={`option-button ${selectedOptimizations.includes(opt) ? 'selected' : ''}`}
              onClick={() => handleOptimizationToggle(opt)}
            >
              {opt}
            </button>
          ))}
        </div>
      </div>

      <div className="benchmark-section">
        <h3>Upload Test Image</h3>
        <FileUpload
          onFileUpload={handleFileUpload}
          accept="image/*"
          loading={loading}
        />
        {uploadedFile && (
          <p className="file-name">Selected: {uploadedFile.name}</p>
        )}
      </div>

      <button
        className="run-benchmark-button"
        onClick={handleRunBenchmark}
        disabled={loading || !uploadedFile || selectedModels.length === 0 || selectedOptimizations.length === 0}
      >
        {loading ? (
          <>
            <Loader className="spinner" size={20} />
            Running Benchmark...
          </>
        ) : (
          'Run Benchmark'
        )}
      </button>

      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}

      {result && result.benchmark_results && (
        <div className="benchmark-results">
          <h3>Benchmark Results</h3>

          <div className="charts-container">
            <div className="chart-wrapper">
              <h4>Frames Per Second (Higher is Better)</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="FPS" fill="#667eea" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-wrapper">
              <h4>Inference Time (Lower is Better)</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Inference Time (ms)" fill="#764ba2" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="results-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Optimization</th>
                  <th>Avg Time (ms)</th>
                  <th>FPS</th>
                  <th>Min Time (ms)</th>
                  <th>Max Time (ms)</th>
                  <th>Std Dev (ms)</th>
                </tr>
              </thead>
              <tbody>
                {result.benchmark_results.map((r, idx) => (
                  <tr key={idx} className={r.error ? 'error-row' : ''}>
                    <td>{r.model}</td>
                    <td>{r.optimization}</td>
                    {r.error ? (
                      <td colSpan="5" className="error-cell">{r.error}</td>
                    ) : (
                      <>
                        <td>{(r.avg_inference_time * 1000).toFixed(2)}</td>
                        <td>{r.fps.toFixed(2)}</td>
                        <td>{(r.min_time * 1000).toFixed(2)}</td>
                        <td>{(r.max_time * 1000).toFixed(2)}</td>
                        <td>{(r.std_time * 1000).toFixed(2)}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default BenchmarkView;
