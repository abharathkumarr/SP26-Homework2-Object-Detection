"""
Roboflow Dataset Download Script

This script downloads a dataset from Roboflow for testing object detection models.
The dataset is used ONLY for evaluation (not training).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import shutil


def setup_roboflow():
    """Install and import roboflow"""
    try:
        from roboflow import Roboflow
        return Roboflow
    except ImportError:
        print("📦 Installing roboflow package...")
        os.system("pip install roboflow")
        from roboflow import Roboflow
        return Roboflow


def download_dataset(api_key, workspace, project, version, output_dir="data"):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
        output_dir: Output directory
    """
    Roboflow = setup_roboflow()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔑 Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    print(f"📂 Accessing workspace: {workspace}")
    print(f"📂 Accessing project: {project}")
    project_obj = rf.workspace(workspace).project(project)
    
    print(f"📥 Downloading version {version} in COCO format...")
    dataset = project_obj.version(version).download("coco", location=str(output_path / "roboflow_download"))
    
    print(f"✅ Dataset downloaded to: {dataset.location}")
    
    # Organize files
    organize_dataset(Path(dataset.location), output_path)
    
    return dataset


def organize_dataset(download_path, output_path):
    """
    Organize downloaded dataset into standard structure
    
    Expected structure after organization:
    data/
      images/
        train/
        valid/
        test/
      annotations/
        train.json
        valid.json
        test.json
    """
    print("\n📁 Organizing dataset...")
    
    # Create directories
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    # Find all splits (train, valid, test)
    for split in ["train", "valid", "test"]:
        split_dir = download_path / split
        if not split_dir.exists():
            continue
        
        print(f"  Processing {split} split...")
        
        # Copy images
        split_images_dir = images_dir / split
        split_images_dir.mkdir(exist_ok=True)
        
        image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        for img in image_files:
            shutil.copy2(img, split_images_dir / img.name)
        
        # Copy annotations
        annotation_file = split_dir / "_annotations.coco.json"
        if annotation_file.exists():
            shutil.copy2(annotation_file, annotations_dir / f"{split}.json")
        
        print(f"    ✅ {len(image_files)} images copied")
    
    print("✅ Dataset organization complete!")
    print(f"\n📊 Dataset location:")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_dir}")


def get_dataset_info(annotations_dir):
    """Get information about the downloaded dataset"""
    annotations_dir = Path(annotations_dir)
    
    info = {}
    for split in ["train", "valid", "test"]:
        annotation_file = annotations_dir / f"{split}.json"
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                info[split] = {
                    "num_images": len(data.get("images", [])),
                    "num_annotations": len(data.get("annotations", [])),
                    "categories": data.get("categories", [])
                }
    
    return info


def print_dataset_summary(output_dir):
    """Print summary of downloaded dataset"""
    annotations_dir = Path(output_dir) / "annotations"
    info = get_dataset_info(annotations_dir)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for split, split_info in info.items():
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {split_info['num_images']}")
        print(f"  Annotations: {split_info['num_annotations']}")
        if split == "train":  # Print categories only once
            print(f"\n  Categories:")
            for cat in split_info['categories']:
                print(f"    - {cat['name']} (ID: {cat['id']})")
    
    print("\n" + "="*60)


def get_popular_datasets():
    """Return list of popular datasets"""
    return {
        "coco-sample": {
            "workspace": "microsoft",
            "project": "coco-dataset",
            "version": 1,
            "description": "COCO sample dataset with multiple object categories"
        },
        "vehicles": {
            "workspace": "roboflow-100",
            "project": "vehicles-q0vsj",
            "version": 2,
            "description": "Vehicle detection dataset"
        },
        "people": {
            "workspace": "roboflow-100",
            "project": "pedestrian-detection",
            "version": 1,
            "description": "People/pedestrian detection"
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Download dataset from Roboflow for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with API key
  python download_roboflow_dataset.py --api-key YOUR_KEY --workspace myworkspace --project myproject --version 1
  
  # Use popular dataset
  python download_roboflow_dataset.py --api-key YOUR_KEY --preset coco-sample
  
  # List available presets
  python download_roboflow_dataset.py --list-presets

Get your API key from: https://app.roboflow.com/settings/api
        """
    )
    
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--workspace', type=str, help='Workspace name')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--version', type=int, help='Dataset version')
    parser.add_argument('--preset', type=str, help='Use popular dataset preset')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        print("\n📋 Available Dataset Presets:\n")
        for name, info in get_popular_datasets().items():
            print(f"  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Workspace: {info['workspace']}")
            print(f"    Project: {info['project']}")
            print(f"    Version: {info['version']}\n")
        return
    
    # Check API key
    if not args.api_key:
        print("❌ Error: API key is required!")
        print("\nGet your API key from: https://app.roboflow.com/settings/api")
        print("\nThen run:")
        print("  python download_roboflow_dataset.py --api-key YOUR_KEY --preset coco-sample")
        sys.exit(1)
    
    # Use preset or custom dataset
    if args.preset:
        presets = get_popular_datasets()
        if args.preset not in presets:
            print(f"❌ Error: Unknown preset '{args.preset}'")
            print(f"\nAvailable presets: {', '.join(presets.keys())}")
            print("\nRun with --list-presets to see details")
            sys.exit(1)
        
        preset = presets[args.preset]
        workspace = preset['workspace']
        project = preset['project']
        version = preset['version']
        print(f"📦 Using preset: {args.preset}")
        print(f"   {preset['description']}\n")
    else:
        # Custom dataset
        if not all([args.workspace, args.project, args.version]):
            print("❌ Error: --workspace, --project, and --version are required for custom datasets")
            print("\nOr use --preset with a popular dataset:")
            print("  python download_roboflow_dataset.py --api-key YOUR_KEY --preset coco-sample")
            sys.exit(1)
        
        workspace = args.workspace
        project = args.project
        version = args.version
    
    # Download
    try:
        dataset = download_dataset(
            api_key=args.api_key,
            workspace=workspace,
            project=project,
            version=version,
            output_dir=args.output
        )
        
        # Print summary
        print_dataset_summary(args.output)
        
        print("\n✅ Dataset ready for use!")
        print("\n📝 Next Steps:")
        print("  1. Run inference: python scripts/run_inference.py")
        print("  2. Evaluate: python evaluation/evaluate.py --gt data/annotations/valid.json --pred results/predictions.json")
        print("  3. Benchmark: python evaluation/benchmark.py --image_dir data/images/valid")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Verify workspace/project/version exist")
        print("  3. Check internet connection")
        sys.exit(1)


if __name__ == '__main__':
    main()
