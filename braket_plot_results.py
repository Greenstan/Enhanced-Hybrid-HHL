"""
Plotting functions for AWS Braket HHL Circuit Depth Experiment

Visualizes circuit depth, execution time, fidelity, and scaling behavior.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_braket_experiment_results(complete_results_file, show_plots=True, save_plots=True):
    """
    Create comprehensive visualization of Braket HHL experiment results.
    
    Args:
        complete_results_file: Path to complete experiment results JSON file
        show_plots: Whether to display plots
        save_plots: Whether to save plots to file
        
    Returns:
        Dictionary with aggregated statistics
    """
    # Load results
    with open(complete_results_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    params = data['parameters']
    matrix_sizes = params['matrix_sizes']
    num_trials = params['num_trials']
    
    # Aggregate results by matrix size
    results_by_size = {size: [] for size in matrix_sizes}
    
    for submission, retrieval in zip(data['submission_results'], data['retrieval_results']):
        if 'error' in submission or 'error' in retrieval:
            continue
        
        size = submission['matrix_size']
        
        results_by_size[size].append({
            'circuit_depth': submission['circuit_depth'],
            'num_projections': submission['num_projections'],
            'fidelity': retrieval['fidelity'],
            'total_time': retrieval.get('total_problem_time', 0),
            'success_rate': retrieval['average_success_rate'],
            'projection_times': retrieval.get('projection_execution_times', [])
        })
    
    # Calculate statistics for each size
    valid_sizes = []
    avg_depths = []
    std_depths = []
    avg_fidelities = []
    std_fidelities = []
    avg_times = []
    std_times = []
    avg_success_rates = []
    avg_projection_times = []
    std_projection_times = []
    
    for size in matrix_sizes:
        trials = results_by_size[size]
        if len(trials) == 0:
            continue
        
        valid_sizes.append(size)
        
        depths = [t['circuit_depth'] for t in trials]
        fidelities = [t['fidelity'] for t in trials]
        times = [t['total_time'] for t in trials if t['total_time'] > 0]
        success_rates = [t['success_rate'] for t in trials]
        
        # Collect all individual projection times for this matrix size
        all_projection_times = []
        for t in trials:
            if t['projection_times']:
                all_projection_times.extend([pt for pt in t['projection_times'] if pt is not None])
        
        avg_depths.append(np.mean(depths))
        std_depths.append(np.std(depths))
        avg_fidelities.append(np.mean(fidelities))
        std_fidelities.append(np.std(fidelities))
        avg_success_rates.append(np.mean(success_rates))
        
        if times:
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))
        else:
            avg_times.append(0)
            std_times.append(0)
        
        if all_projection_times:
            avg_projection_times.append(np.mean(all_projection_times))
            std_projection_times.append(np.std(all_projection_times))
        else:
            avg_projection_times.append(0)
            std_projection_times.append(0)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 8))
    
    # Plot 1: Circuit Depth vs Matrix Size
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(valid_sizes, avg_depths, 'o-', linewidth=2, markersize=8, color='blue', label='Avg Circuit Depth')
    ax1.fill_between(valid_sizes, 
                      [d - s for d, s in zip(avg_depths, std_depths)],
                      [d + s for d, s in zip(avg_depths, std_depths)],
                      alpha=0.2, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('Matrix Size (N)', fontsize=11)
    ax1.set_ylabel('Circuit Depth', fontsize=11)
    ax1.set_title('HHL Circuit Depth vs Matrix Size', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Total Execution Time vs Matrix Size
    ax2 = plt.subplot(2, 3, 2)
    if any(t > 0 for t in avg_times):
        ax2.plot(valid_sizes, avg_times, 'o-', linewidth=2, markersize=8, color='green', label='Avg Total Time')
        ax2.fill_between(valid_sizes,
                          [max(0, t - s) for t, s in zip(avg_times, std_times)],
                          [t + s for t, s in zip(avg_times, std_times)],
                          alpha=0.2, color='green', label='±1 Std Dev')
        
        # Add T values as scatter points (T = 4500 corresponds to 1s)
        # T should be 1.3x higher than the total problem execution time
        # Convert T back to time in seconds: time = T / 4500
        T_values = [15*4500, 30*4500, 50*4500, 120*4500]  
        T_times = [T / ( 4500 ) for T in T_values]  # Y-axis values for T points
        
        ax2.scatter(valid_sizes, T_times, s=50, color='brown', marker='D', 
                   label='T values (for TLP)', zorder=5)
        
        # Add T value labels above each point
        for i, (size, T_time, T) in enumerate(zip(valid_sizes, T_times, T_values)):
            ax2.annotate(f'T={int(T)}', 
                        xy=(size, T_time), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.set_ylabel('Total Execution Time (seconds)', fontsize=11)
        # Set y-axis limit to add padding for T labels
        max_y = max(max(T_times), max(avg_times)) * 1.15
        ax2.set_ylim([0, max_y])
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Timing data\nnot available', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel('Matrix Size (N)', fontsize=11)
    ax2.set_title('Total Problem Execution Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual Projection Measurement Time vs Matrix Size
    ax3 = plt.subplot(2, 3, 4)
    if any(t > 0 for t in avg_projection_times):
        ax3.plot(valid_sizes, avg_projection_times, 'o-', linewidth=2, markersize=8, color='purple', label='Avg Projection Time')
        ax3.fill_between(valid_sizes,
                          [max(0, t - s) for t, s in zip(avg_projection_times, std_projection_times)],
                          [t + s for t, s in zip(avg_projection_times, std_projection_times)],
                          alpha=0.2, color='purple', label='±1 Std Dev')
        ax3.set_ylabel('Individual Projection Time (seconds)', fontsize=11)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Projection timing\ndata not available', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlabel('Matrix Size (N)', fontsize=11)
    ax3.set_title('Individual Projection Measurement Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fidelity vs Matrix Size
    ax4 = plt.subplot(2, 3, 5)
    ax4.plot(valid_sizes, avg_fidelities, 'o-', linewidth=2, markersize=8, color='orange', label='Avg Fidelity')
    ax4.fill_between(valid_sizes,
                      [max(0, f - s) for f, s in zip(avg_fidelities, std_fidelities)],
                      [min(1, f + s) for f, s in zip(avg_fidelities, std_fidelities)],
                      alpha=0.2, color='orange', label='±1 Std Dev')
    ax4.set_xlabel('Matrix Size (N)', fontsize=11)
    ax4.set_ylabel('Fidelity', fontsize=11)
    ax4.set_title('Solution Fidelity vs Matrix Size', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Statistics Summary (spanning right column, both rows)
    ax_summary = plt.subplot(2, 3, (3, 6))
    ax_summary.axis('off')
    
    # Calculate statistics
    min_depth = min(avg_depths)
    max_depth = max(avg_depths)
    avg_all_depths = np.mean(avg_depths)
    min_fidelity = min(avg_fidelities)
    max_fidelity = max(avg_fidelities)
    avg_all_fidelities = np.mean(avg_fidelities)
    
    # Growth rate estimation
    if len(valid_sizes) > 2:
        slope, intercept, r_value, p_value, std_err = linregress(valid_sizes, avg_depths)
        growth_rate = slope
        r_squared = r_value**2
        
        if any(t > 0 for t in avg_times):
            slope_time, _, r_value_time, _, _ = linregress(valid_sizes, avg_times)
            growth_rate_time = slope_time
            r_squared_time = r_value_time**2
        else:
            growth_rate_time = None
            r_squared_time = None
    else:
        growth_rate = None
        r_squared = None
        growth_rate_time = None
        r_squared_time = None
    
    stats_text = f"""
EXPERIMENT SUMMARY
{'='*35}

Device: {params['device_arn']}
Region: {params['aws_region']}
Preprocessing: {params['preprocessing_mode']}
Clock Qubits: {params['k_qubits']}
Shots: {params['shots']}

Matrix Sizes: {len(valid_sizes)}
Size Range: {min(valid_sizes)} to {max(valid_sizes)}
Trials per Size: {num_trials}
Successful: {sum(len(results_by_size[s]) for s in valid_sizes)}

CIRCUIT DEPTH:
  Min: {min_depth:.0f}
  Max: {max_depth:.0f}
  Avg: {avg_all_depths:.1f}
  Range: {max_depth - min_depth:.0f}

FIDELITY:
  Min: {min_fidelity:.4f}
  Max: {max_fidelity:.4f}
  Avg: {avg_all_fidelities:.4f}

Execution Time:
 Min: {min(avg_times):.4f} s
 Max: {max(avg_times):.4f} s
 Avg: {np.mean([t for t in avg_times if t > 0]):.4f} s
"""

    
#     if growth_rate is not None:
#         stats_text += f"""
# GROWTH ANALYSIS:
#   Depth slope: {growth_rate:.2f}/N
#   Depth R²: {r_squared:.4f}
# """
#         if growth_rate_time:
#             stats_text += f"  Time slope: {growth_rate_time:.4f}s/N\n  Time R²: {r_squared_time:.4f}\n"
    
    ax_summary.text(0.05, 0.95, stats_text, transform=ax_summary.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'AWS Braket HHL Circuit Depth Experiment Results', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_plots:
        output_filename = complete_results_file.replace('.json', '_plots2.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {output_filename}")
    
    if show_plots:
        plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT ANALYSIS")
    print("="*80)
    print(f"Smallest matrix (N={min(valid_sizes)}): Depth = {avg_depths[0]:.0f}, Fidelity = {avg_fidelities[0]:.4f}")
    print(f"Largest matrix (N={max(valid_sizes)}):  Depth = {avg_depths[-1]:.0f}, Fidelity = {avg_fidelities[-1]:.4f}")
    print(f"Depth increase: {avg_depths[-1]/avg_depths[0]:.2f}x")
    if any(t > 0 for t in avg_times):
        print(f"Time increase: {avg_times[-1]/avg_times[0]:.2f}x")
    print(f"Average circuit depth: {avg_all_depths:.1f}")
    print(f"Average fidelity: {avg_all_fidelities:.4f}")
    print("="*80)
    
    return {
        'valid_sizes': valid_sizes,
        'avg_depths': avg_depths,
        'std_depths': std_depths,
        'avg_fidelities': avg_fidelities,
        'std_fidelities': std_fidelities,
        'avg_times': avg_times,
        'std_times': std_times,
        'avg_projection_times': avg_projection_times,
        'std_projection_times': std_projection_times,
        'avg_success_rates': avg_success_rates,
        'growth_rate': growth_rate,
        'r_squared': r_squared
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python braket_plot_results.py <complete_results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    stats = plot_braket_experiment_results(results_file, show_plots=True, save_plots=True)
    
    print("\n✓ Analysis complete!")
