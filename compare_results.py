#!/usr/bin/env python3
"""
Script to compare results from different checkpoint experiments.
Reads dreamer_results_rank_0.json files from various experiment directories
and provides comprehensive comparison analysis.
"""

import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import argparse
import yaml


def find_result_files(base_dir="outputs", date_filter=None, variant_filter=None):
    """
    Find all dreamer_results_rank_0.json files in the output directory structure.

    Args:
        base_dir: Base directory to search in
        date_filter: Optional date filter (e.g., "2025-09-24" to filter by specific date)
        variant_filter: Optional variant filter (e.g., "baseline", "PT", "PT-FT")

    Returns:
        List of tuples: (experiment_path, result_file_path, checkpoint_info)
    """
    result_files = []

    # Pattern to match the result files
    pattern = os.path.join(base_dir, "**/dreamer_results_rank_0.json")

    for result_file in glob.glob(pattern, recursive=True):
        # Extract experiment identifier from path
        path_parts = Path(result_file).parts

        # Find the experiment directory (e.g., "2025-09-24/01-11-21")
        exp_dir = None
        checkpoint_info = None
        for i, part in enumerate(path_parts):
            if part.startswith("2025-") and i + 1 < len(path_parts):
                exp_dir = f"{part}/{path_parts[i+1]}"
                # Look for checkpoint info in the path
                for j in range(i+2, len(path_parts)):
                    if path_parts[j].startswith("epoch="):
                        checkpoint_info = path_parts[j]
                        break
                break

        if exp_dir:
            # Apply date filter if specified
            if date_filter and not exp_dir.startswith(date_filter):
                continue

            # Apply variant filter if specified
            if variant_filter:
                variant = extract_variant_info(exp_dir, base_dir)
                if variant_filter.lower() not in variant.lower():
                    continue

            result_files.append((exp_dir, result_file, checkpoint_info))

    return sorted(result_files)


def extract_variant_info(exp_name, base_dir="outputs"):
    """Extract variant information from Hydra config."""
    config_path = os.path.join(base_dir, exp_name, ".hydra", "config.yaml")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract vision model variant
        vision_variant = config.get('model', {}).get('vision_model', {}).get('variant', '')

        # Extract a short name from the variant path
        if vision_variant:
            # Extract the last meaningful part of the path
            if 'InternVL3-1B-hf-no_seminit' in vision_variant:
                return 'baseline'
            elif 'InternVL3-1B-HF_0ms-PT-FT' in vision_variant:
                return 'PT-FT'
            elif 'InternVL3-1B-HF_0ms' in vision_variant:
                return 'PT'
            else:
                # Extract the last directory name
                return os.path.basename(vision_variant)

        # Fallback to experiment name from config
        return config.get('name', 'unknown')

    except Exception as e:
        print(f"Warning: Could not read config for {exp_name}: {e}")
        return 'unknown'


def extract_epoch_step_info(checkpoint_info):
    """Extract epoch and step information from checkpoint string."""
    if not checkpoint_info:
        return "unknown", "unknown"

    try:
        # Parse "epoch=000-step=000500.ckpt" format
        parts = checkpoint_info.replace('.ckpt', '').split('-')
        epoch = parts[0].split('=')[1] if len(parts) > 0 and '=' in parts[0] else "unknown"
        step = parts[1].split('=')[1] if len(parts) > 1 and '=' in parts[1] else "unknown"
        return epoch, step
    except Exception:
        return "unknown", "unknown"


def load_results(result_files):
    """
    Load all result files and organize them into a structured format.

    Args:
        result_files: List of (experiment_path, result_file_path, checkpoint_info) tuples

    Returns:
        Dictionary with experiment data
    """
    experiments = {}

    for exp_name, file_path, checkpoint_info in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract epoch and step info
            epoch, step = extract_epoch_step_info(checkpoint_info)

            # Add metadata
            data['experiment'] = exp_name
            data['file_path'] = file_path
            data['variant'] = extract_variant_info(exp_name)
            data['epoch'] = epoch
            data['step'] = step
            data['checkpoint_info'] = checkpoint_info or "unknown"

            experiments[exp_name] = data
            print(f"Loaded results for experiment: {exp_name} (variant: {data['variant']}, epoch: {epoch}, step: {step})")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return experiments


def create_comparison_dataframe(experiments):
    """
    Create a pandas DataFrame for easy comparison and analysis.

    Args:
        experiments: Dictionary of experiment data

    Returns:
        pandas.DataFrame with comparison data
    """
    rows = []

    for exp_name, data in experiments.items():
        row = {
            'experiment': exp_name,
            'variant': data.get('variant', 'unknown'),
            'epoch': data.get('epoch', 'unknown'),
            'step': data.get('step', 'unknown'),
            'total_success_rate': data.get('success_rate_total_instruction', 0),
            'crash_success_rate': data.get('success_rate_instruction_crash', 0),
            'target_speed_success_rate': data.get('success_rate_instruction_target_speed', 0),
            'lane_change_success_rate': data.get('success_rate_instruction_lane_change', 0),
            'slower_success_rate': data.get('success_rate_instruction_slower', 0),
            'stop_success_rate': data.get('success_rate_instruction_stop', 0),
            'faster_success_rate': data.get('success_rate_instruction_faster', 0),
            'num_samples': data.get('num_samples_instruction', 0)
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(df):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*100)
    
    # Format the dataframe for better display
    display_df = df.copy()
    
    # Convert success rates to percentages
    rate_columns = [col for col in df.columns if 'success_rate' in col]
    for col in rate_columns:
        display_df[col] = (display_df[col] * 100).round(2)
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        'variant': 'Variant',
        'epoch': 'Epoch',
        'step': 'Step',
        'total_success_rate': 'Total (%)',
        'crash_success_rate': 'Crash (%)',
        'target_speed_success_rate': 'Target Speed (%)',
        'lane_change_success_rate': 'Lane Change (%)',
        'slower_success_rate': 'Slower (%)',
        'stop_success_rate': 'Stop (%)',
        'faster_success_rate': 'Faster (%)',
        'num_samples': 'Samples'
    })
    
    print(display_df.to_string(index=False, float_format='%.2f'))
    print("="*100)


def create_visualizations(df, output_dir="comparison_plots"):
    """Create visualization plots for the comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall success rate comparison with variants
    plt.figure(figsize=(15, 8))

    # Create colors based on variants
    unique_variants = df['variant'].unique()
    colors = sns.color_palette("Set3", len(unique_variants))
    variant_colors = {variant: colors[i] for i, variant in enumerate(unique_variants)}
    bar_colors = [variant_colors[variant] for variant in df['variant']]

    bars = plt.bar(range(len(df)), df['total_success_rate'] * 100, color=bar_colors)
    plt.xlabel('Experiment')
    plt.ylabel('Total Success Rate (%)')
    plt.title('Overall Success Rate Comparison by Variant')

    # Create experiment labels with variant info
    exp_labels = [f"{exp}\n({variant})" for exp, variant in zip(df['experiment'], df['variant'])]
    plt.xticks(range(len(df)), exp_labels, rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')

    # Add legend for variants
    legend_elements = [mpatches.Rectangle((0,0),1,1, facecolor=variant_colors[variant], label=variant)
                      for variant in unique_variants]
    plt.legend(handles=legend_elements, title='Variants', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_success_rates.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed breakdown by instruction type with variants
    instruction_types = ['crash_success_rate', 'target_speed_success_rate',
                        'lane_change_success_rate', 'slower_success_rate',
                        'stop_success_rate', 'faster_success_rate']

    instruction_labels = ['Crash', 'Target Speed', 'Lane Change', 'Slower', 'Stop', 'Faster']

    plt.figure(figsize=(18, 10))

    x = range(len(df))
    width = 0.12

    for i, (col, label) in enumerate(zip(instruction_types, instruction_labels)):
        offset = (i - len(instruction_types)/2) * width
        plt.bar([xi + offset for xi in x], df[col] * 100, width, label=label)

    plt.xlabel('Experiment')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Instruction Type and Variant')

    # Create experiment labels with variant and epoch info
    exp_labels = [f"{exp}\n({variant}, E{epoch})" for exp, variant, epoch in
                  zip(df['experiment'], df['variant'], df['epoch'])]
    plt.xticks(x, exp_labels, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of success rates with variant information
    plt.figure(figsize=(14, 8))
    heatmap_data = df[instruction_types].T * 100

    # Create column labels with variant and epoch info
    col_labels = [f"{exp}\n({variant}, E{epoch})" for exp, variant, epoch in
                  zip(df['experiment'], df['variant'], df['epoch'])]
    heatmap_data.columns = col_labels
    heatmap_data.index = instruction_labels

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Success Rate (%)'})
    plt.title('Success Rate Heatmap by Instruction Type and Variant')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_detailed_comparison(experiments, df, output_file="comparison_plots/detailed_comparison.json"):
    """Save detailed comparison data to JSON file."""
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'summary_statistics': {
            'num_experiments': len(experiments),
            'best_overall': df.loc[df['total_success_rate'].idxmax(), 'experiment'],
            'worst_overall': df.loc[df['total_success_rate'].idxmin(), 'experiment'],
            'mean_success_rate': df['total_success_rate'].mean(),
            'std_success_rate': df['total_success_rate'].std()
        },
        'detailed_results': experiments,
        'comparison_table': df.to_dict('records')
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nDetailed comparison saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--base-dir', default='outputs',
                       help='Base directory to search for results (default: outputs)')
    parser.add_argument('--output-dir', default='comparison_plots',
                       help='Directory to save plots (default: comparison_plots)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--date', help='Filter by date (e.g., 2025-09-24)')
    parser.add_argument('--variant', help='Filter by variant (e.g., baseline, PT, PT-FT)')

    args = parser.parse_args()

    print("Searching for result files...")
    if args.date:
        print(f"Filtering by date: {args.date}")
    if args.variant:
        print(f"Filtering by variant: {args.variant}")
    result_files = find_result_files(args.base_dir, args.date, args.variant)
    
    if not result_files:
        print(f"No result files found in {args.base_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Load all results
    experiments = load_results(result_files)
    
    if not experiments:
        print("No valid experiments loaded")
        return
    
    # Create comparison dataframe
    df = create_comparison_dataframe(experiments)
    
    # Print comparison table
    print_comparison_table(df)
    
    # Create visualizations
    if not args.no_plots:
        print(f"\nGenerating visualizations in {args.output_dir}...")
        create_visualizations(df, args.output_dir)
    
    # Save detailed comparison
    save_detailed_comparison(experiments, df, f"{args.output_dir}/detailed_comparison.json")
    
    print(f"\nComparison complete! Found {len(experiments)} experiments.")


if __name__ == "__main__":
    main()
