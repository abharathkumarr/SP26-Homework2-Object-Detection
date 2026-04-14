"""
Annotation Tool for Object Detection

This script provides a simple GUI tool for annotating images for object detection.
Use this to create ground truth annotations for evaluation.
"""

import cv2
import json
import os
from pathlib import Path
import argparse


class AnnotationTool:
    def __init__(self, image_dir, output_file):
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.images = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        self.current_idx = 0
        self.annotations = self.load_annotations()
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.boxes = []
        self.current_box = None
        self.current_class = 0
        
        # COCO classes (subset for demo)
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
        
        print("Annotation Tool Controls:")
        print("- Click and drag to draw bounding box")
        print("- Press 'n' for next image")
        print("- Press 'p' for previous image")
        print("- Press 's' to save annotations")
        print("- Press 'c' to change class (current: {})".format(self.classes[self.current_class]))
        print("- Press 'd' to delete last box")
        print("- Press 'q' to quit")
    
    def load_annotations(self):
        """Load existing annotations if available"""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return {"images": [], "annotations": []}
    
    def save_annotations(self):
        """Save annotations to file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"✅ Annotations saved to {self.output_file}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum box size
                    self.boxes.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: x, y, width, height
                        'class_id': self.current_class,
                        'class_name': self.classes[self.current_class]
                    })
                    print(f"Added box: {self.classes[self.current_class]}")
                
                self.start_point = None
                self.current_box = None
    
    def draw_boxes(self, image):
        """Draw all boxes on image"""
        img = image.copy()
        
        # Draw saved boxes
        for box in self.boxes:
            x, y, w, h = box['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = box['class_name']
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return img
    
    def run(self):
        """Main annotation loop"""
        if not self.images:
            print("❌ No images found in directory")
            return
        
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        while True:
            if self.current_idx >= len(self.images):
                break
            
            image_path = self.images[self.current_idx]
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"⚠️ Could not load image: {image_path}")
                self.current_idx += 1
                continue
            
            # Check if this image already has annotations
            image_id = str(image_path.name)
            existing_annot = None
            for img_annot in self.annotations.get('images', []):
                if img_annot['file_name'] == image_id:
                    existing_annot = img_annot
                    break
            
            if existing_annot and not self.boxes:
                # Load existing boxes
                for ann in self.annotations.get('annotations', []):
                    if ann['image_id'] == existing_annot['id']:
                        self.boxes.append({
                            'bbox': ann['bbox'],
                            'class_id': ann['category_id'],
                            'class_name': self.classes[ann['category_id']]
                        })
            
            # Display image with boxes
            display_img = self.draw_boxes(image)
            
            # Add info text
            info_text = f"Image {self.current_idx + 1}/{len(self.images)} | Class: {self.classes[self.current_class]} | Boxes: {len(self.boxes)}"
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow('Annotation Tool', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.save_annotations()
                break
            
            elif key == ord('n'):
                # Save current annotations and move to next
                self.save_current_image()
                self.boxes = []
                self.current_idx += 1
            
            elif key == ord('p'):
                # Save and go to previous
                self.save_current_image()
                self.boxes = []
                self.current_idx = max(0, self.current_idx - 1)
            
            elif key == ord('s'):
                self.save_current_image()
                self.save_annotations()
            
            elif key == ord('c'):
                # Change class
                self.current_class = (self.current_class + 1) % len(self.classes)
                print(f"Current class: {self.classes[self.current_class]}")
            
            elif key == ord('d'):
                # Delete last box
                if self.boxes:
                    deleted = self.boxes.pop()
                    print(f"Deleted box: {deleted['class_name']}")
        
        cv2.destroyAllWindows()
    
    def save_current_image(self):
        """Save annotations for current image"""
        if not self.boxes:
            return
        
        image_path = self.images[self.current_idx]
        image_id = str(image_path.name)
        
        # Remove existing annotations for this image
        self.annotations['images'] = [
            img for img in self.annotations.get('images', [])
            if img['file_name'] != image_id
        ]
        
        # Read image to get dimensions
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        # Add image info
        img_info = {
            'id': len(self.annotations['images']),
            'file_name': image_id,
            'width': width,
            'height': height
        }
        self.annotations['images'].append(img_info)
        
        # Remove old annotations for this image
        self.annotations['annotations'] = [
            ann for ann in self.annotations.get('annotations', [])
            if ann['image_id'] != img_info['id']
        ]
        
        # Add new annotations
        for box in self.boxes:
            ann = {
                'id': len(self.annotations['annotations']),
                'image_id': img_info['id'],
                'category_id': box['class_id'],
                'bbox': box['bbox'],
                'area': box['bbox'][2] * box['bbox'][3],
                'iscrowd': 0
            }
            self.annotations['annotations'].append(ann)
        
        print(f"✅ Saved {len(self.boxes)} boxes for {image_id}")


def main():
    parser = argparse.ArgumentParser(description='Annotation Tool for Object Detection')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images to annotate')
    parser.add_argument('--output', type=str, default='annotations/annotations.json',
                       help='Output file for annotations (COCO format)')
    
    args = parser.parse_args()
    
    tool = AnnotationTool(args.image_dir, args.output)
    tool.run()


if __name__ == '__main__':
    main()
