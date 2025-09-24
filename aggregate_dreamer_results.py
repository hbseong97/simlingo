#!/usr/bin/env python3
"""
Script to aggregate dreamer_results from rank 0 to 7
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_dreamer_results(predictions_dir, ranks=range(8)):
    """Load dreamer_results files for specified ranks"""
    results = {}
    
    for rank in ranks:
        file_path = predictions_dir / f"dreamer_results_rank_{rank}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[rank] = json.load(f)
            print(f"Loaded rank {rank}: {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return results

def aggregate_results(results_dict):
    """Aggregate results across all ranks"""
    # Initialize aggregated metrics
    aggregated = defaultdict(list)
    total_samples = 0
    
    # Collect all metrics from each rank
    for rank, data in results_dict.items():
        total_samples += data.get('num_samples_instruction', 0)
        
        for key, value in data.items():
            if key != 'num_samples_instruction' and isinstance(value, (int, float)):
                aggregated[key].append(value)
    
    # Calculate weighted averages and overall statistics
    final_results = {
        'total_samples_across_all_ranks': total_samples,
        'num_ranks': len(results_dict),
        'individual_rank_results': results_dict
    }
    
    # Calculate simple averages across ranks
    for metric, values in aggregated.items():
        if values:
            final_results[f'{metric}_mean_across_ranks'] = sum(values) / len(values)
            final_results[f'{metric}_min_across_ranks'] = min(values)
            final_results[f'{metric}_max_across_ranks'] = max(values)
    
    # Calculate weighted average for total instruction success rate
    total_successful = 0
    total_samples_for_weighting = 0
    
    for rank, data in results_dict.items():
        samples = data.get('num_samples_instruction', 0)
        success_rate = data.get('success_rate_total_instruction', 0)
        total_successful += success_rate * samples
        total_samples_for_weighting += samples
    
    if total_samples_for_weighting > 0:
        final_results['success_rate_total_instruction_weighted'] = total_successful / total_samples_for_weighting
    
    return final_results

def main():
    # Define the predictions directory
    predictions_dir = Path("outputs/2025-09-23/13-12-28/predictions")
    
    print("=== Aggregating Dreamer Results (Rank 0-7) ===")
    print(f"Looking in directory: {predictions_dir}")
    
    # Load all dreamer results
    results = load_dreamer_results(predictions_dir)
    
    if not results:
        print("No dreamer_results files found!")
        return
    
    print(f"\nLoaded {len(results)} rank files")
    
    # Aggregate the results
    aggregated = aggregate_results(results)
    
    # Save aggregated results
    output_file = predictions_dir / "dreamer_results_aggregated.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n=== Aggregated Results ===")
    print(f"Total samples across all ranks: {aggregated['total_samples_across_all_ranks']}")
    print(f"Number of ranks: {aggregated['num_ranks']}")
    
    if 'success_rate_total_instruction_weighted' in aggregated:
        print(f"Weighted overall success rate: {aggregated['success_rate_total_instruction_weighted']:.4f}")
    
    print(f"\n=== Success Rate Statistics Across Ranks ===")
    for key in sorted(aggregated.keys()):
        if key.endswith('_mean_across_ranks') and 'success_rate' in key:
            metric_name = key.replace('_mean_across_ranks', '')
            mean_val = aggregated[key]
            min_val = aggregated.get(f'{metric_name}_min_across_ranks', 'N/A')
            max_val = aggregated.get(f'{metric_name}_max_across_ranks', 'N/A')
            print(f"{metric_name}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Also create a summary table
    print(f"\n=== Per-Rank Summary ===")
    print("Rank | Samples | Total Success Rate")
    print("-" * 35)
    for rank in sorted(results.keys()):
        data = results[rank]
        samples = data.get('num_samples_instruction', 0)
        success_rate = data.get('success_rate_total_instruction', 0)
        print(f"{rank:4d} | {samples:7d} | {success_rate:14.4f}")

if __name__ == "__main__":
    main()
