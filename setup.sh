#!/bin/bash

# Setup script for Object Detection Inference System

echo "🚀 Setting up Object Detection Inference System..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install evaluation dependencies
echo ""
echo "📦 Installing evaluation dependencies..."
cd evaluation
pip install -r requirements.txt
cd ..

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/results
mkdir -p backend/models
mkdir -p data/images
mkdir -p data/videos
mkdir -p data/annotations
mkdir -p results

# Download sample models
echo ""
echo "📥 Downloading sample models..."
cd backend/models
python3 << EOF
from ultralytics import YOLO
print("Downloading YOLOv8n...")
model = YOLO('yolov8n.pt')
print("✓ YOLOv8n downloaded")
EOF
cd ../..

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
if command -v npm &> /dev/null; then
    cd frontend
    echo "Installing npm dependencies..."
    npm install
    cd ..
    echo "✓ Frontend dependencies installed"
else
    echo "⚠️  npm not found. Please install Node.js to set up the frontend."
    echo "   Visit: https://nodejs.org/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo ""
echo "1. Start the backend:"
echo "   cd backend"
echo "   python app/main.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "3. For Google Colab:"
echo "   Upload colab_demo.ipynb to Google Colab"
echo ""
echo "📚 Documentation: See README.md for detailed instructions"
echo ""
