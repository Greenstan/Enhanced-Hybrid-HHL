"""
AWS Braket HHL Clock Qubit Comparison Plots

Creates plots comparing execution times and circuit depths for different 
matrix sizes (N) and different clock qubit values (k).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


def plot_k_comparison(complete_results_file, save_plots=True, show_plots=False):
    """
    Create comparison plots for execution time and circuit depth across k values.
    
    Args:
        complete_results_file: Path to the complete results JSON file
        save_plots: Whether to save plots to file
        show_plots: Whether to display plots interactively
    """
    
    # Load results
    with open(complete_results_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    params = data['parameters']
    matrix_sizes = sorted(params['matrix_sizes'])
    k_values = sorted(params['k_qubits'])
    
    # Organize data by N and k
    data_by_config = defaultdict(lambda: defaultdict(list))
    
    for retrieval in data.get('retrieval_results', []):
        if 'error' in retrieval:
            continue
        
        # Find corresponding submission to get N and k
        submission_file = retrieval['submission_file']
        submission = None
        for sub in data['submission_results']:
            if sub.get('submission_file') == submission_file:
                submission = sub
                break
        
        if not submission:
            continue
        
        N = submission['matrix_size']
        k = submission['k_qubits']
        
        # Collect metrics
        data_by_config[N][k].append({
            'total_time': retrieval.get('total_problem_time', 0),
            'circuit_depth': retrieval['circuit_depth'],
            'fidelity': retrieval.get('fidelity', 0),
        })
    
    # Prepare data for plotting
    avg_times = defaultdict(dict)
    std_times = defaultdict(dict)
    avg_depths = defaultdict(dict)
    std_depths = defaultdict(dict)
    
    for N in matrix_sizes:
        for k in k_values:
            if k in data_by_config[N]:
                times = [t['total_time'] for t in data_by_config[N][k] if t['total_time'] > 0]
                depths = [t['circuit_depth'] for t in data_by_config[N][k]]
                
                if times:
                    avg_times[k][N] = np.mean(times)
                    std_times[k][N] = np.std(times)
                # Don't set to 0, just skip if no data
                
                if depths:
                    avg_depths[k][N] = np.mean(depths)
                    std_depths[k][N] = np.std(depths)
    
    # Define colors for different k values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Execution Time vs Matrix Size for different k values
    for idx, k in enumerate(k_values):
        N_vals = []
        time_vals = []
        time_errs = []
        
        for N in matrix_sizes:
            if N in avg_times[k]:  # Only plot if we have data
                N_vals.append(N)
                time_vals.append(avg_times[k][N])
                time_errs.append(std_times[k][N])
        
        if N_vals:
            # Convert to arrays for easier manipulation
            N_vals = np.array(N_vals)
            time_vals = np.array(time_vals)
            time_errs = np.array(time_errs)
            
            # Plot line and markers
            ax1.plot(N_vals, time_vals, 
                    marker='o', markersize=8, linewidth=2,
                    color=colors[idx], label=f'k={k}')
            
            # Add shaded error region
            # ax1.fill_between(N_vals, 
            #                time_vals - time_errs, 
            #                time_vals + time_errs,
            #                color=colors[idx], alpha=0.2)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time vs Matrix Size for Different Clock Qubits (k)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_xticks(matrix_sizes)
    
    # Plot 2: Circuit Depth vs Matrix Size for different k values
    for idx, k in enumerate(k_values):
        N_vals = []
        depth_vals = []
        depth_errs = []
        
        for N in matrix_sizes:
            if N in avg_depths[k]:
                N_vals.append(N)
                depth_vals.append(avg_depths[k][N])
                depth_errs.append(std_depths[k][N])
        
        if N_vals:
            # Convert to arrays
            N_vals = np.array(N_vals)
            depth_vals = np.array(depth_vals)
            depth_errs = np.array(depth_errs)
            
            # Plot line and markers
            ax2.plot(N_vals, depth_vals,
                    marker='o', markersize=8, linewidth=2,
                    color=colors[idx], label=f'k={k}')
            
            # # Add shaded error region
            # ax2.fill_between(N_vals,
            #                depth_vals - depth_errs,
            #                depth_vals + depth_errs,
            #                color=colors[idx], alpha=0.2)
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax2.set_title('Circuit Depth vs Matrix Size for Different Clock Qubits (k)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_xticks(matrix_sizes)
    
    # Overall title
    plt.suptitle(f'AWS Braket HHL - Clock Qubit Comparison\n' + 
                 f'Device: {params["device_arn"]}, Preprocessing: {params["preprocessing_mode"]}, ' +
                 f'Shots: {params["shots"]}',
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_plots:
        output_filename = complete_results_file.replace('.json', '_k_comparison_plots3.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {output_filename}")
    
    if show_plots:
        plt.show()
    
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python braket_plot_k_comparison.py <complete_results_file.json> [--show]")
        sys.exit(1)
    
    complete_results_file = sys.argv[1]
    show_plots = '--show' in sys.argv
    
    print(f"\nGenerating clock qubit comparison plots from: {complete_results_file}\n")
    plot_k_comparison(complete_results_file, save_plots=True, show_plots=show_plots)
    print("\n✓ Plot generation complete!")
