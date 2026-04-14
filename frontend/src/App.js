import React, { useState, useCallback } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ModelSelector from './components/ModelSelector';
import ResultDisplay from './components/ResultDisplay';
import BenchmarkView from './components/BenchmarkView';
import { detectImage, detectVideo, benchmarkModels } from './services/api';
import { Upload, Zap, BarChart3, Loader } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('detect');
  const [selectedModel, setSelectedModel] = useState('yolov8n');
  const [selectedOptimization, setSelectedOptimization] = useState('none');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const fileType = file.type.startsWith('video') ? 'video' : 'image';
      
      let response;
      if (fileType === 'image') {
        response = await detectImage(
          file,
          selectedModel,
          selectedOptimization,
          confThreshold,
          iouThreshold
        );
      } else {
        response = await detectVideo(
          file,
          selectedModel,
          selectedOptimization,
          confThreshold,
          iouThreshold
        );
      }

      setResult({ ...response, fileType });
    } catch (err) {
      setError(err.message || 'An error occurred during detection');
    } finally {
      setLoading(false);
    }
  }, [selectedModel, selectedOptimization, confThreshold, iouThreshold]);

  const handleBenchmark = useCallback(async (file, models, optimizations) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await benchmarkModels(file, models, optimizations);
      setResult({ ...response, fileType: 'benchmark' });
    } catch (err) {
      setError(err.message || 'An error occurred during benchmarking');
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="App">
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <Zap className="icon" />
            Object Detection System
          </h1>
          <p className="subtitle">Optimized Inference with Multiple Models</p>
        </div>
      </header>

      <div className="container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'detect' ? 'active' : ''}`}
            onClick={() => setActiveTab('detect')}
          >
            <Upload size={20} />
            Detection
          </button>
          <button
            className={`tab ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            <BarChart3 size={20} />
            Benchmark
          </button>
        </div>

        <div className="content">
          {activeTab === 'detect' ? (
            <>
              <ModelSelector
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                selectedOptimization={selectedOptimization}
                setSelectedOptimization={setSelectedOptimization}
                confThreshold={confThreshold}
                setConfThreshold={setConfThreshold}
                iouThreshold={iouThreshold}
                setIouThreshold={setIouThreshold}
              />

              <FileUpload
                onFileUpload={handleFileUpload}
                accept="image/*,video/*"
                loading={loading}
              />

              {loading && (
                <div className="loading">
                  <Loader className="spinner" />
                  <p>Processing... This may take a moment</p>
                </div>
              )}

              {error && (
                <div className="error">
                  <p>{error}</p>
                </div>
              )}

              {result && result.fileType !== 'benchmark' && (
                <ResultDisplay result={result} />
              )}
            </>
          ) : (
            <BenchmarkView
              onBenchmark={handleBenchmark}
              loading={loading}
              result={result}
              error={error}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
