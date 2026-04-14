"""
Run Inference on Dataset

This script runs object detection inference on all images in a dataset
and saves the results in COCO format for evaluation.
"""

import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import argparse

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from ultralytics import YOLO
import cv2


def run_inference(image_dir, output_file, model_name="yolov8n", optimization="none"):
    """
    Run inference on all images in a directory
    
    Args:
        image_dir: Directory containing images
        output_file: Output file for predictions (COCO format)
        model_name: Model to use
        optimization: Optimization backend
    """
    image_dir = Path(image_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"📦 Loading model: {model_name}")
    if optimization == "onnx":
        model_path = f"{model_name}.onnx"
        if not Path(model_path).exists():
            print(f"  Exporting to ONNX...")
            base_model = YOLO(f"{model_name}.pt")
            base_model.export(format="onnx", dynamic=True, simplify=True)
        model = YOLO(model_path)
    else:
        model = YOLO(f"{model_name}.pt")
    
    print(f"✅ Model loaded\n")
    
    # Get all images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"📸 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        return
    
    # Run inference
    predictions = []
    inference_times = []
    
    print(f"\n🔄 Running inference...")
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not read {img_path.name}")
            continue
        
        height, width = img.shape[:2]
        
        # Run inference
        start_time = time.time()
        results = model.predict(img_path, conf=0.25, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Parse results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Convert to COCO format: [x, y, width, height]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                predictions.append({
                    "image_id": img_path.name,
                    "category_id": cls,
                    "bbox": bbox,
                    "score": conf
                })
    
    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    fps = 1.0 / avg_time
    
    print(f"\n✅ Inference complete!")
    print(f"   Total images: {len(image_files)}")
    print(f"   Total detections: {len(predictions)}")
    print(f"   Avg inference time: {avg_time*1000:.2f}ms")
    print(f"   FPS: {fps:.2f}")
    
    # Save predictions
    output_data = {
        "predictions": predictions,
        "metadata": {
            "model": model_name,
            "optimization": optimization,
            "num_images": len(image_files),
            "num_predictions": len(predictions),
            "avg_inference_time": avg_time,
            "fps": fps
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Predictions saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on dataset')
    parser.add_argument('--image_dir', type=str, default='data/images/valid',
                       help='Directory with images')
    parser.add_argument('--output', type=str, default='results/predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--model', type=str, default='yolov8n',
                       help='Model name (yolov8n, yolov8s, etc.)')
    parser.add_argument('--optimization', type=str, default='none',
                       choices=['none', 'onnx'],
                       help='Optimization backend')
    
    args = parser.parse_args()
    
    run_inference(
        image_dir=args.image_dir,
        output_file=args.output,
        model_name=args.model,
        optimization=args.optimization
    )
    
    print("\n📝 Next step:")
    print(f"  python evaluation/evaluate.py --gt data/annotations/valid.json --pred {args.output}")


if __name__ == '__main__':
    main()
