from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
import uvicorn
import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.model_manager import ModelManager
from utils.video_processor import VideoProcessor
from utils.image_processor import ImageProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection API",
    description="Optimized object detection inference API supporting multiple models and optimization backends",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Initialize model manager
model_manager = ModelManager()

# Initialize processors
video_processor = VideoProcessor()
image_processor = ImageProcessor()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting Object Detection API...")
    print("📦 Loading models...")
    
    # Load default models
    try:
        model_manager.load_model("yolov8n", optimization="none")
        print("✅ YOLOv8n loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load YOLOv8n: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "detect_image": "/detect/image",
            "detect_video": "/detect/video",
            "benchmark": "/benchmark"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.loaded_models),
        "timestamp": time.time()
    }


@app.get("/models")
async def list_models():
    """List available models and optimization backends"""
    return {
        "available_models": model_manager.get_available_models(),
        "loaded_models": list(model_manager.loaded_models.keys()),
        "optimization_backends": ["none", "onnx", "tensorrt", "torchscript"]
    }


@app.post("/models/load")
async def load_model(
    model_name: str = Form(...),
    optimization: str = Form("none")
):
    """Load a specific model with optimization"""
    try:
        success = model_manager.load_model(model_name, optimization)
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} loaded with {optimization} optimization",
                "model_key": f"{model_name}_{optimization}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model_name: str = Form("yolov8n"),
    optimization: str = Form("none"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """Perform object detection on an image"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get model
        model_key = f"{model_name}_{optimization}"
        if model_key not in model_manager.loaded_models:
            model_manager.load_model(model_name, optimization)
        
        model_info = model_manager.loaded_models[model_key]
        
        # Run inference
        start_time = time.time()
        results = image_processor.process(
            str(file_path),
            model_info["model"],
            conf_threshold,
            iou_threshold
        )
        inference_time = time.time() - start_time
        
        # Save result image
        result_filename = f"result_{file.filename}"
        result_path = RESULT_DIR / result_filename
        image_processor.save_visualization(results, str(result_path))
        
        return {
            "status": "success",
            "detections": results["detections"],
            "inference_time": inference_time,
            "fps": 1.0 / inference_time,
            "model": model_name,
            "optimization": optimization,
            "result_image": f"/results/{result_filename}",
            "num_detections": len(results["detections"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    model_name: str = Form("yolov8n"),
    optimization: str = Form("none"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    skip_frames: int = Form(1)
):
    """Perform object detection on a video"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get model
        model_key = f"{model_name}_{optimization}"
        if model_key not in model_manager.loaded_models:
            model_manager.load_model(model_name, optimization)
        
        model_info = model_manager.loaded_models[model_key]
        
        # Process video
        result_filename = f"result_{file.filename}"
        result_path = RESULT_DIR / result_filename
        
        results = video_processor.process(
            str(file_path),
            str(result_path),
            model_info["model"],
            conf_threshold,
            iou_threshold,
            skip_frames
        )
        
        return {
            "status": "success",
            "total_frames": results["total_frames"],
            "processed_frames": results["processed_frames"],
            "average_inference_time": results["avg_inference_time"],
            "average_fps": results["avg_fps"],
            "model": model_name,
            "optimization": optimization,
            "result_video": f"/results/{result_filename}",
            "detections_summary": results["detections_summary"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Retrieve a result file"""
    file_path = RESULT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post("/benchmark")
async def benchmark_models(
    test_image: UploadFile = File(...),
    models: List[str] = Form(["yolov8n"]),
    optimizations: List[str] = Form(["none", "onnx"]),
    num_runs: int = Form(100)
):
    """Benchmark multiple models and optimization backends"""
    try:
        # Save test image
        file_path = UPLOAD_DIR / test_image.filename
        with open(file_path, "wb") as f:
            content = await test_image.read()
            f.write(content)
        
        benchmark_results = []
        
        for model_name in models:
            for optimization in optimizations:
                try:
                    # Load model
                    model_key = f"{model_name}_{optimization}"
                    if model_key not in model_manager.loaded_models:
                        model_manager.load_model(model_name, optimization)
                    
                    model_info = model_manager.loaded_models[model_key]
                    
                    # Warm-up
                    for _ in range(10):
                        image_processor.process(str(file_path), model_info["model"])
                    
                    # Benchmark
                    times = []
                    for _ in range(num_runs):
                        start = time.time()
                        image_processor.process(str(file_path), model_info["model"])
                        times.append(time.time() - start)
                    
                    avg_time = sum(times) / len(times)
                    
                    benchmark_results.append({
                        "model": model_name,
                        "optimization": optimization,
                        "avg_inference_time": avg_time,
                        "fps": 1.0 / avg_time,
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_time": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                    })
                    
                except Exception as e:
                    benchmark_results.append({
                        "model": model_name,
                        "optimization": optimization,
                        "error": str(e)
                    })
        
        return {
            "status": "success",
            "benchmark_results": benchmark_results,
            "num_runs": num_runs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
