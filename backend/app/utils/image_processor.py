import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path


class ImageProcessor:
    """Process images for object detection"""
    
    def __init__(self):
        self.colors = self._generate_colors(80)  # COCO has 80 classes
    
    def _generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        return colors.tolist()
    
    def process(self, image_path: str, model, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict:
        """
        Process an image with object detection
        
        Args:
            image_path: Path to image
            model: Detection model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Dict with detections and annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": result.names[cls]
                })
        
        # Annotate image
        annotated_image = self._annotate_image(image.copy(), detections)
        
        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "original_image": image
        }
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on image"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            class_name = det["class_name"]
            class_id = det["class_id"]
            
            # Get color
            color = self.colors[class_id % len(self.colors)]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return image
    
    def save_visualization(self, results: Dict, output_path: str):
        """Save annotated image"""
        cv2.imwrite(output_path, results["annotated_image"])
