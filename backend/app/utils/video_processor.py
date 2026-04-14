import cv2
import numpy as np
from typing import Dict, List
import time
from tqdm import tqdm


class VideoProcessor:
    """Process videos for object detection"""
    
    def __init__(self):
        self.colors = self._generate_colors(80)
    
    def _generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        return colors.tolist()
    
    def process(
        self,
        video_path: str,
        output_path: str,
        model,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        skip_frames: int = 1
    ) -> Dict:
        """
        Process a video with object detection
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            model: Detection model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            skip_frames: Process every Nth frame
        
        Returns:
            Dict with processing statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        inference_times = []
        all_detections = []
        frame_idx = 0
        processed_frames = 0
        
        print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % skip_frames == 0:
                # Run inference
                start_time = time.time()
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Parse and annotate
                detections = self._parse_results(results)
                all_detections.extend(detections)
                frame = self._annotate_frame(frame, detections)
                processed_frames += 1
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        # Calculate statistics
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Detection statistics
        class_counts = {}
        for det in all_detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "avg_inference_time": avg_inference_time,
            "avg_fps": avg_fps,
            "total_detections": len(all_detections),
            "detections_summary": class_counts
        }
    
    def _parse_results(self, results) -> List[Dict]:
        """Parse detection results"""
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
        return detections
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            class_name = det["class_name"]
            class_id = det["class_id"]
            
            # Get color
            color = self.colors[class_id % len(self.colors)]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return frame
