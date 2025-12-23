"""
AWS Braket HHL Clock Qubit Comparison Table

Tabulates and compares execution times for different matrix sizes (N) 
and different clock qubit values (k) from complete experiment results.
"""

import json
import numpy as np
import pandas as pd
import sys
from collections import defaultdict


def tabulate_k_comparison(complete_results_file):
    """
    Create tables comparing execution times across different k values for each matrix size.
    
    Args:
        complete_results_file: Path to the complete results JSON file
    """
    
    # Load results
    with open(complete_results_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    params = data['parameters']
    matrix_sizes = params['matrix_sizes']
    k_values = params['k_qubits']
    
    print("="*100)
    print("AWS BRAKET HHL CIRCUIT DEPTH EXPERIMENT - CLOCK QUBIT COMPARISON")
    print("="*100)
    print(f"\nExperiment Type: {data['experiment_type']}")
    print(f"Matrix Sizes (N): {matrix_sizes}")
    print(f"Clock Qubits (k): {k_values}")
    print(f"Trials per configuration: {params['num_trials']}")
    print(f"Preprocessing: {params['preprocessing_mode']}")
    print(f"Device: {params['device_arn']}")
    print(f"Shots: {params['shots']}")
    
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
            'avg_success_rate': retrieval.get('average_success_rate', 0),
            'num_projections': retrieval['num_projections'],
            'projection_times': retrieval.get('projection_execution_times', [])
        })
    
    # Print summary table for each matrix size
    print("\n" + "="*100)
    print("SUMMARY: EXECUTION TIME vs CLOCK QUBITS (k) FOR EACH MATRIX SIZE (N)")
    print("="*100)
    
    # Create DataFrame for summary statistics
    summary_data = []
    for N in sorted(matrix_sizes):
        for k in sorted(k_values):
            if k not in data_by_config[N]:
                continue
            
            trials = data_by_config[N][k]
            times = [t['total_time'] for t in trials if t['total_time'] > 0]
            depths = [t['circuit_depth'] for t in trials]
            fidelities = [t['fidelity'] for t in trials if t['fidelity'] > 0]
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
            else:
                avg_time = std_time = min_time = max_time = 0
            
            avg_depth = np.mean(depths) if depths else 0
            avg_fidelity = np.mean(fidelities) if fidelities else 0
            
            summary_data.append({
                'N': N,
                'k': k,
                'Trials': len(trials),
                'Avg Time (s)': avg_time,
                'Std Time (s)': std_time,
                'Min Time (s)': min_time,
                'Max Time (s)': max_time,
                'Avg Depth': avg_depth,
                'Avg Fidelity': avg_fidelity
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary for each N
    for N in sorted(matrix_sizes):
        print(f"\n{'─'*100}")
        print(f"MATRIX SIZE N = {N}")
        print(f"{'─'*100}")
        
        n_df = summary_df[summary_df['N'] == N].drop('N', axis=1)
        if n_df.empty:
            print("  No data available")
        else:
            print("\n" + n_df.to_string(index=False))
    
    # Create pivot tables for comparison
    print("\n" + "="*100)
    print("COMPARISON TABLE: AVERAGE EXECUTION TIME (seconds)")
    print("="*100)
    
    time_pivot = summary_df.pivot(index='k', columns='N', values='Avg Time (s)')
    print("\n" + time_pivot.to_string())
    
    # Print circuit depth comparison
    print("\n" + "="*100)
    print("COMPARISON TABLE: AVERAGE CIRCUIT DEPTH")
    print("="*100)
    
    depth_pivot = summary_df.pivot(index='k', columns='N', values='Avg Depth')
    print("\n" + depth_pivot.to_string())
    
    # Print fidelity comparison
    print("\n" + "="*100)
    print("COMPARISON TABLE: AVERAGE FIDELITY")
    print("="*100)
    
    fidelity_pivot = summary_df.pivot(index='k', columns='N', values='Avg Fidelity')
    print("\n" + fidelity_pivot.to_string())
    
    # Calculate and print relative speedup/slowdown
    print("\n" + "="*100)
    baseline_k = min(k_values) if k_values else None
    print(f"RELATIVE TIME COMPARISON (normalized to k={baseline_k})")
    print("="*100)
    
    if baseline_k:
        relative_data = []
        for k in sorted(k_values):
            row_data = {'k': k}
            for N in sorted(matrix_sizes):
                baseline_time = time_pivot.loc[baseline_k, N] if baseline_k in time_pivot.index and N in time_pivot.columns else None
                current_time = time_pivot.loc[k, N] if k in time_pivot.index and N in time_pivot.columns else None
                
                if baseline_time and current_time and baseline_time > 0:
                    row_data[N] = current_time / baseline_time
                else:
                    row_data[N] = np.nan
            relative_data.append(row_data)
        
        relative_df = pd.DataFrame(relative_data).set_index('k')
        print("\n" + relative_df.to_string())
    
    # Print detailed statistics for each configuration
    print("\n" + "="*100)
    print("DETAILED STATISTICS")
    print("="*100)
    
    detailed_data = []
    for N in sorted(matrix_sizes):
        for k in sorted(k_values):
            if k not in data_by_config[N]:
                continue
            
            trials = data_by_config[N][k]
            times = [t['total_time'] for t in trials if t['total_time'] > 0]
            depths = [t['circuit_depth'] for t in trials]
            fidelities = [t['fidelity'] for t in trials if t['fidelity'] > 0]
            success_rates = [t['avg_success_rate'] for t in trials if t['avg_success_rate'] > 0]
            
            stats = {
                'N': N,
                'k': k,
                'Trials': len(trials)
            }
            
            if times:
                stats['Time Mean'] = np.mean(times)
                stats['Time Std'] = np.std(times)
                stats['Time Min'] = np.min(times)
                stats['Time Max'] = np.max(times)
            
            if depths:
                stats['Depth Mean'] = np.mean(depths)
                stats['Depth Std'] = np.std(depths)
                stats['Depth Min'] = np.min(depths)
                stats['Depth Max'] = np.max(depths)
            
            if fidelities:
                stats['Fidelity Mean'] = np.mean(fidelities)
                stats['Fidelity Std'] = np.std(fidelities)
                stats['Fidelity Min'] = np.min(fidelities)
                stats['Fidelity Max'] = np.max(fidelities)
            
            if success_rates:
                stats['Success Rate Mean'] = np.mean(success_rates)
                stats['Success Rate Std'] = np.std(success_rates)
            
            detailed_data.append(stats)
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Display grouped by N
    for N in sorted(matrix_sizes):
        n_detail = detailed_df[detailed_df['N'] == N]
        if not n_detail.empty:
            print(f"\n{'─'*100}")
            print(f"MATRIX SIZE N = {N}")
            print(f"{'─'*100}")
            print("\n" + n_detail.drop('N', axis=1).to_string(index=False))
    
    # Save DataFrames to CSV files
    output_dir = '/'.join(complete_results_file.split('/')[:-1])
    base_name = complete_results_file.split('/')[-1].replace('.json', '')
    
    summary_csv = f"{output_dir}/{base_name}_summary.csv"
    time_pivot_csv = f"{output_dir}/{base_name}_time_comparison.csv"
    depth_pivot_csv = f"{output_dir}/{base_name}_depth_comparison.csv"
    fidelity_pivot_csv = f"{output_dir}/{base_name}_fidelity_comparison.csv"
    detailed_csv = f"{output_dir}/{base_name}_detailed_stats.csv"
    
    summary_df.to_csv(summary_csv, index=False)
    time_pivot.to_csv(time_pivot_csv)
    depth_pivot.to_csv(depth_pivot_csv)
    fidelity_pivot.to_csv(fidelity_pivot_csv)
    detailed_df.to_csv(detailed_csv, index=False)
    
    print("\n" + "="*100)
    print("CSV FILES SAVED")
    print("="*100)
    print(f"  Summary:              {summary_csv}")
    print(f"  Time Comparison:      {time_pivot_csv}")
    print(f"  Depth Comparison:     {depth_pivot_csv}")
    print(f"  Fidelity Comparison:  {fidelity_pivot_csv}")
    print(f"  Detailed Statistics:  {detailed_csv}")
    
    print("\n" + "="*100)
    print("END OF COMPARISON")
    print("="*100)
    
    return {
        'summary': summary_df,
        'time_comparison': time_pivot,
        'depth_comparison': depth_pivot,
        'fidelity_comparison': fidelity_pivot,
        'detailed_stats': detailed_df
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python braket_tabulate_k_comparison.py <complete_results_file.json>")
        sys.exit(1)
    
    complete_results_file = sys.argv[1]
    dataframes = tabulate_k_comparison(complete_results_file)
    
    # DataFrames are returned and can be used for further analysis
    # dataframes['summary'], dataframes['time_comparison'], etc.
