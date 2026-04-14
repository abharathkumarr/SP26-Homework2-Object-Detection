"""
Benchmark Script for Model Performance

This script runs comprehensive benchmarks comparing different models
and optimization backends on speed and accuracy.
"""

import time
import json
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'backend' / 'app'))

from models.model_manager import ModelManager
from utils.image_processor import ImageProcessor


class Benchmark:
    def __init__(self, image_dir, models, optimizations):
        """
        Initialize benchmark
        
        Args:
            image_dir: Directory with test images
            models: List of model names to benchmark
            optimizations: List of optimization backends to test
        """
        self.image_dir = Path(image_dir)
        self.models = models
        self.optimizations = optimizations
        
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        
        # Get test images
        self.test_images = list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.png'))
        
        if not self.test_images:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"📊 Benchmark Setup:")
        print(f"   Models: {models}")
        print(f"   Optimizations: {optimizations}")
        print(f"   Test Images: {len(self.test_images)}")
    
    def benchmark_model(self, model_name, optimization, num_warmup=10, num_runs=100):
        """Benchmark a single model configuration"""
        print(f"\n🔄 Benchmarking {model_name} with {optimization}...")
        
        # Load model
        model_key = f"{model_name}_{optimization}"
        if model_key not in self.model_manager.loaded_models:
            success = self.model_manager.load_model(model_name, optimization)
            if not success:
                return None
        
        model_info = self.model_manager.loaded_models[model_key]
        model = model_info['model']
        
        # Select a test image
        test_image = str(self.test_images[0])
        
        # Warmup
        print(f"   Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.image_processor.process(test_image, model)
        
        # Benchmark
        print(f"   Running benchmark ({num_runs} runs)...")
        times = []
        
        for i in range(num_runs):
            start = time.time()
            results = self.image_processor.process(test_image, model)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i + 1}/{num_runs}")
        
        # Compute statistics
        times = np.array(times)
        
        result = {
            'model': model_name,
            'optimization': optimization,
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'p95_time': float(np.percentile(times, 95)),
            'p99_time': float(np.percentile(times, 99)),
            'fps': float(1.0 / np.mean(times)),
            'num_runs': num_runs
        }
        
        print(f"   ✅ Avg: {result['avg_time']*1000:.2f}ms, FPS: {result['fps']:.2f}")
        
        return result
    
    def run_all(self, num_warmup=10, num_runs=100):
        """Run all benchmarks"""
        all_results = []
        
        for model_name in self.models:
            for optimization in self.optimizations:
                try:
                    result = self.benchmark_model(
                        model_name,
                        optimization,
                        num_warmup,
                        num_runs
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    all_results.append({
                        'model': model_name,
                        'optimization': optimization,
                        'error': str(e)
                    })
        
        return all_results
    
    def print_results(self, results):
        """Print benchmark results in a nice table"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Header
        print(f"{'Model':<15} {'Optimization':<15} {'Avg (ms)':<12} {'FPS':<10} {'P95 (ms)':<12}")
        print("-"*80)
        
        # Sort by FPS (descending)
        valid_results = [r for r in results if 'error' not in r]
        valid_results.sort(key=lambda x: x.get('fps', 0), reverse=True)
        
        for result in valid_results:
            print(f"{result['model']:<15} "
                  f"{result['optimization']:<15} "
                  f"{result['avg_time']*1000:<12.2f} "
                  f"{result['fps']:<10.2f} "
                  f"{result['p95_time']*1000:<12.2f}")
        
        # Print errors
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print("\nErrors:")
            for result in error_results:
                print(f"  {result['model']} ({result['optimization']}): {result['error']}")
        
        print("="*80)
        
        # Speed comparison
        if len(valid_results) >= 2:
            fastest = valid_results[0]
            slowest = valid_results[-1]
            speedup = slowest['avg_time'] / fastest['avg_time']
            
            print(f"\n📈 Performance Summary:")
            print(f"   Fastest: {fastest['model']} ({fastest['optimization']}) - {fastest['fps']:.2f} FPS")
            print(f"   Slowest: {slowest['model']} ({slowest['optimization']}) - {slowest['fps']:.2f} FPS")
            print(f"   Speedup: {speedup:.2f}x")
    
    def save_results(self, results, output_file):
        """Save results to JSON file"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Object Detection Models')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory with test images')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['yolov8n', 'yolov8s'],
                       help='Models to benchmark')
    parser.add_argument('--optimizations', type=str, nargs='+',
                       default=['none', 'onnx'],
                       help='Optimization backends to test')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    benchmark = Benchmark(args.image_dir, args.models, args.optimizations)
    results = benchmark.run_all(args.warmup, args.runs)
    benchmark.print_results(results)
    benchmark.save_results(results, args.output)


if __name__ == '__main__':
    main()
