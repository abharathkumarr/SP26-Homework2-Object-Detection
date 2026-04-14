"""
Evaluation Script for Object Detection

This script calculates COCO-style mAP metrics for object detection results
compared to ground truth annotations.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


class COCOEvaluator:
    def __init__(self, gt_file, pred_file):
        """
        Initialize evaluator with ground truth and prediction files
        
        Args:
            gt_file: Path to ground truth annotations (COCO format)
            pred_file: Path to prediction results (COCO format)
        """
        self.gt_file = Path(gt_file)
        self.pred_file = Path(pred_file)
        
        # Load annotations
        with open(self.gt_file, 'r') as f:
            self.gt_data = json.load(f)
        
        with open(self.pred_file, 'r') as f:
            self.pred_data = json.load(f)
        
        # Build image id mapping
        self.image_id_map = {img['file_name']: img['id'] 
                            for img in self.gt_data.get('images', [])}
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        Boxes in format: [x, y, width, height]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        # Compute intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_ap(self, recalls, precisions):
        """Compute Average Precision using 11-point interpolation"""
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    def evaluate_class(self, class_id, iou_threshold=0.5):
        """Evaluate a single class"""
        # Get ground truth boxes for this class
        gt_boxes = defaultdict(list)
        for ann in self.gt_data.get('annotations', []):
            if ann['category_id'] == class_id:
                gt_boxes[ann['image_id']].append(ann)
        
        # Get predictions for this class
        pred_boxes = defaultdict(list)
        for pred in self.pred_data.get('predictions', []):
            if pred['category_id'] == class_id:
                pred_boxes[pred['image_id']].append(pred)
        
        # Sort predictions by confidence (descending)
        all_preds = []
        for img_id, preds in pred_boxes.items():
            for pred in preds:
                all_preds.append((img_id, pred))
        all_preds.sort(key=lambda x: x[1].get('score', 1.0), reverse=True)
        
        # Compute TP, FP for each prediction
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        # Track which ground truth boxes have been matched
        matched_gt = defaultdict(set)
        
        for idx, (img_id, pred) in enumerate(all_preds):
            max_iou = 0
            max_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt in enumerate(gt_boxes.get(img_id, [])):
                if gt_idx in matched_gt[img_id]:
                    continue
                
                iou = self.compute_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # Check if it's a true positive
            if max_iou >= iou_threshold and max_gt_idx >= 0:
                tp[idx] = 1
                matched_gt[img_id].add(max_gt_idx)
            else:
                fp[idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        num_gt = sum(len(boxes) for boxes in gt_boxes.values())
        
        recalls = tp_cumsum / max(num_gt, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = self.compute_ap(recalls, precisions)
        
        return {
            'ap': ap,
            'num_gt': num_gt,
            'num_pred': len(all_preds),
            'tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
            'fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
        }
    
    def evaluate(self, iou_thresholds=[0.5, 0.75]):
        """Run full evaluation"""
        # Get all category IDs
        category_ids = set()
        for ann in self.gt_data.get('annotations', []):
            category_ids.add(ann['category_id'])
        
        results = {}
        
        for iou_threshold in iou_thresholds:
            class_results = {}
            aps = []
            
            for class_id in sorted(category_ids):
                class_result = self.evaluate_class(class_id, iou_threshold)
                class_results[class_id] = class_result
                aps.append(class_result['ap'])
            
            mean_ap = np.mean(aps) if aps else 0.0
            
            results[f'IoU={iou_threshold}'] = {
                'mAP': mean_ap,
                'class_results': class_results
            }
        
        # Compute mAP@[0.5:0.95] (COCO standard)
        iou_range = np.linspace(0.5, 0.95, 10)
        all_aps = []
        
        for iou_threshold in iou_range:
            aps = []
            for class_id in sorted(category_ids):
                class_result = self.evaluate_class(class_id, iou_threshold)
                aps.append(class_result['ap'])
            all_aps.append(np.mean(aps) if aps else 0.0)
        
        results['mAP@[0.5:0.95]'] = np.mean(all_aps)
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("COCO-Style Evaluation Results")
        print("="*60)
        
        print(f"\nmAP@[0.5:0.95]: {results['mAP@[0.5:0.95]']:.4f}")
        
        for metric_name, metric_data in results.items():
            if metric_name == 'mAP@[0.5:0.95]':
                continue
            
            print(f"\n{metric_name}:")
            print(f"  mAP: {metric_data['mAP']:.4f}")
            
            print("\n  Per-class results:")
            for class_id, class_result in metric_data['class_results'].items():
                print(f"    Class {class_id}:")
                print(f"      AP: {class_result['ap']:.4f}")
                print(f"      GT boxes: {class_result['num_gt']}")
                print(f"      Predictions: {class_result['num_pred']}")
                print(f"      TP: {class_result['tp']}, FP: {class_result['fp']}")
        
        print("\n" + "="*60)


def convert_detections_to_coco(detections_file, output_file):
    """
    Convert detections from API format to COCO format
    
    Args:
        detections_file: JSON file with detections from the API
        output_file: Output file in COCO format
    """
    with open(detections_file, 'r') as f:
        detections = json.load(f)
    
    predictions = []
    
    for detection in detections:
        pred = {
            'image_id': detection.get('image_id', 0),
            'category_id': detection.get('class_id', 0),
            'bbox': detection.get('bbox', []),
            'score': detection.get('confidence', 1.0)
        }
        predictions.append(pred)
    
    output = {'predictions': predictions}
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Converted {len(predictions)} predictions to COCO format")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Object Detection Results')
    parser.add_argument('--gt', type=str, required=True,
                       help='Ground truth annotations file (COCO format)')
    parser.add_argument('--pred', type=str, required=True,
                       help='Prediction results file (COCO format)')
    parser.add_argument('--iou', type=float, nargs='+', default=[0.5, 0.75],
                       help='IoU thresholds for evaluation')
    
    args = parser.parse_args()
    
    evaluator = COCOEvaluator(args.gt, args.pred)
    results = evaluator.evaluate(args.iou)
    evaluator.print_results(results)
    
    # Save results to file
    output_file = Path(args.pred).parent / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()
