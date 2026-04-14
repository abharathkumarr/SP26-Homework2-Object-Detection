from ultralytics import YOLO
from pathlib import Path
import torch
from typing import Dict, Optional
import sys


class ModelManager:
    """Manage multiple object detection models and their optimizations"""
    
    def __init__(self):
        self.loaded_models: Dict = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Device: {self.device}")
        
        # Model configurations
        self.model_configs = {
            "yolov8n": {"type": "yolo", "size": "n", "url": "yolov8n.pt"},
            "yolov8s": {"type": "yolo", "size": "s", "url": "yolov8s.pt"},
            "yolov8m": {"type": "yolo", "size": "m", "url": "yolov8m.pt"},
            "yolov8l": {"type": "yolo", "size": "l", "url": "yolov8l.pt"},
            "yolov8x": {"type": "yolo", "size": "x", "url": "yolov8x.pt"},
            "yolov11n": {"type": "yolo", "size": "n", "url": "yolo11n.pt"},
            "yolov11s": {"type": "yolo", "size": "s", "url": "yolo11s.pt"},
            "yolov11m": {"type": "yolo", "size": "m", "url": "yolo11m.pt"},
        }
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def load_model(self, model_name: str, optimization: str = "none") -> bool:
        """
        Load a model with specified optimization
        
        Args:
            model_name: Name of the model (e.g., 'yolov8n')
            optimization: Optimization backend ('none', 'onnx', 'tensorrt', 'torchscript')
        
        Returns:
            bool: Success status
        """
        try:
            model_key = f"{model_name}_{optimization}"
            
            # Check if already loaded
            if model_key in self.loaded_models:
                print(f"✅ Model {model_key} already loaded")
                return True
            
            # Get model config
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            config = self.model_configs[model_name]
            
            print(f"📦 Loading {model_name} with {optimization} optimization...")
            
            # Load base model
            if config["type"] == "yolo":
                model = YOLO(config["url"])
                
                # Apply optimization
                if optimization == "onnx":
                    model = self._export_onnx(model, model_name)
                elif optimization == "tensorrt":
                    model = self._export_tensorrt(model, model_name)
                elif optimization == "torchscript":
                    model = self._export_torchscript(model, model_name)
            
            # Store model
            self.loaded_models[model_key] = {
                "model": model,
                "config": config,
                "optimization": optimization,
                "device": self.device
            }
            
            print(f"✅ Successfully loaded {model_key}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return False
    
    def _export_onnx(self, model, model_name: str):
        """Export model to ONNX format"""
        try:
            print(f"🔄 Exporting {model_name} to ONNX...")
            onnx_path = f"models/{model_name}.onnx"
            Path("models").mkdir(exist_ok=True)
            
            # Export to ONNX
            model.export(format="onnx", dynamic=True, simplify=True)
            
            # Load ONNX model
            from ultralytics import YOLO
            onnx_model = YOLO(onnx_path)
            
            print(f"✅ ONNX export successful")
            return onnx_model
            
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}, using original model")
            return model
    
    def _export_tensorrt(self, model, model_name: str):
        """Export model to TensorRT format"""
        try:
            if not torch.cuda.is_available():
                print("⚠️ TensorRT requires CUDA, using original model")
                return model
            
            print(f"🔄 Exporting {model_name} to TensorRT...")
            engine_path = f"models/{model_name}.engine"
            Path("models").mkdir(exist_ok=True)
            
            # Export to TensorRT
            model.export(format="engine", device=0, half=True, workspace=4)
            
            # Load TensorRT model
            trt_model = YOLO(engine_path)
            
            print(f"✅ TensorRT export successful")
            return trt_model
            
        except Exception as e:
            print(f"⚠️ TensorRT export failed: {e}, using original model")
            return model
    
    def _export_torchscript(self, model, model_name: str):
        """Export model to TorchScript format"""
        try:
            print(f"🔄 Exporting {model_name} to TorchScript...")
            torchscript_path = f"models/{model_name}.torchscript"
            Path("models").mkdir(exist_ok=True)
            
            # Export to TorchScript
            model.export(format="torchscript")
            
            # Load TorchScript model
            ts_model = YOLO(torchscript_path)
            
            print(f"✅ TorchScript export successful")
            return ts_model
            
        except Exception as e:
            print(f"⚠️ TorchScript export failed: {e}, using original model")
            return model
    
    def get_model(self, model_name: str, optimization: str = "none"):
        """Get a loaded model"""
        model_key = f"{model_name}_{optimization}"
        if model_key not in self.loaded_models:
            self.load_model(model_name, optimization)
        return self.loaded_models[model_key]["model"]
    
    def unload_model(self, model_name: str, optimization: str = "none"):
        """Unload a model to free memory"""
        model_key = f"{model_name}_{optimization}"
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            torch.cuda.empty_cache()
            print(f"🗑️ Unloaded {model_key}")
