import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File } from 'lucide-react';
import './FileUpload.css';

const FileUpload = ({ onFileUpload, accept, loading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept || 'image/*,video/*',
    multiple: false,
    disabled: loading
  });

  return (
    <div className="file-upload-container">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${loading ? 'disabled' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          {isDragActive ? (
            <>
              <Upload size={48} className="upload-icon" />
              <p>Drop the file here...</p>
            </>
          ) : (
            <>
              <File size={48} className="upload-icon" />
              <p className="dropzone-text">
                Drag and drop an image or video here
              </p>
              <p className="dropzone-subtext">or click to select a file</p>
              <button className="upload-button" disabled={loading}>
                Choose File
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
