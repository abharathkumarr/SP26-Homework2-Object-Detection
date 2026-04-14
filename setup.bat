@echo off
REM Setup script for Object Detection Inference System (Windows)

echo 🚀 Setting up Object Detection Inference System...
echo.

REM Check Python version
python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install backend dependencies
echo.
echo 📦 Installing backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

REM Install evaluation dependencies
echo.
echo 📦 Installing evaluation dependencies...
cd evaluation
pip install -r requirements.txt
cd ..

REM Create necessary directories
echo.
echo 📁 Creating directories...
mkdir backend\uploads 2>nul
mkdir backend\results 2>nul
mkdir backend\models 2>nul
mkdir data\images 2>nul
mkdir data\videos 2>nul
mkdir data\annotations 2>nul
mkdir results 2>nul

REM Download sample models
echo.
echo 📥 Downloading sample models...
cd backend\models
python -c "from ultralytics import YOLO; print('Downloading YOLOv8n...'); model = YOLO('yolov8n.pt'); print('✓ YOLOv8n downloaded')"
cd ..\..

REM Frontend setup
echo.
echo 📦 Setting up frontend...
where npm >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd frontend
    echo Installing npm dependencies...
    call npm install
    cd ..
    echo ✓ Frontend dependencies installed
) else (
    echo ⚠️  npm not found. Please install Node.js to set up the frontend.
    echo    Visit: https://nodejs.org/
)

echo.
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo.
echo 1. Start the backend:
echo    cd backend
echo    python app\main.py
echo.
echo 2. In a new terminal, start the frontend:
echo    cd frontend
echo    npm start
echo.
echo 3. For Google Colab:
echo    Upload colab_demo.ipynb to Google Colab
echo.
echo 📚 Documentation: See README.md for detailed instructions
echo.

pause
