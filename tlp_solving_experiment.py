"""
Time-Lock Puzzle Solving Experiment

Runs the TLP lattice puzzle generation and measures solving time for different T values.
Saves results to experiments/tlp_solving directory.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Add puzzle_generation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'puzzle_generation'))
from tlp_lattice import generate_puzzle, solve_puzzle


def run_tlp_solving_experiment(
    T_values=[4500 * 2, 4500 * 15, 4500 * 40, 4500 * 120],
    num_trials=3,
    message=b"This is a secret message!",
    output_dir="./experiments/tlp_solving"
):
    """
    Run TLP solving experiment for different T values.
    
    Args:
        T_values: List of T (time parameter) values to test
        num_trials: Number of trials per T value
        message: Message to encrypt in the puzzle
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    
    print("="*70)
    print("TIME-LOCK PUZZLE SOLVING EXPERIMENT")
    print("="*70)
    print(f"T values: {T_values}")
    print(f"Trials per T: {num_trials}")
    print(f"Message: {message.decode()}")
    print(f"Message length: {len(message)} bytes")
    print("="*70)
    print(f"{'T':<10} {'Trial':<7} {'Gen Time (s)':<15} {'Solve Time (s)':<15} {'Status'}")
    print("-"*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for all results
    all_results = []
    
    for T in T_values:
        for trial in range(num_trials):
            try:
                # Generate puzzle
                gen_start = time.time()
                puzzle = generate_puzzle(T, message)
                gen_end = time.time()
                gen_time = gen_end - gen_start
                
                # Solve puzzle
                solve_start = time.time()
                recovered_message = solve_puzzle(puzzle)
                solve_end = time.time()
                solve_time = solve_end - solve_start
                
                # Verify correctness
                success = (recovered_message == message)
                
                # Store results
                trial_result = {
                    'T': T,
                    'trial': trial,
                    'generation_time': gen_time,
                    'solving_time': solve_time,
                    'total_time': gen_time + solve_time,
                    'success': success,
                    'message_length': len(message),
                    'recovered_message': recovered_message.decode() if success else None,
                    'timestamp': datetime.now().isoformat()
                }
                
                all_results.append(trial_result)
                
                status = '✓' if success else '✗'
                print(f"{T:<10} {trial+1:<7} {gen_time:<15.4f} {solve_time:<15.4f} {status}")
                
            except Exception as e:
                print(f"{T:<10} {trial+1:<7} {'ERROR':<15} {'-':<15} {'✗'}")
                print(f"  Error: {str(e)[:60]}")
                
                trial_result = {
                    'T': T,
                    'trial': trial,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(trial_result)
    
    print("="*70)
    print("Experiment complete!")
    print(f"Successfully completed: {sum(1 for r in all_results if r.get('success'))}/{len(all_results)}")
    print("="*70)
    
    # Calculate statistics per T value
    stats_by_T = {}
    for T in T_values:
        T_results = [r for r in all_results if r.get('T') == T and r.get('success')]
        if T_results:
            gen_times = [r['generation_time'] for r in T_results]
            solve_times = [r['solving_time'] for r in T_results]
            total_times = [r['total_time'] for r in T_results]
            
            stats_by_T[str(T)] = {
                'avg_generation_time': float(np.mean(gen_times)),
                'std_generation_time': float(np.std(gen_times)),
                'avg_solving_time': float(np.mean(solve_times)),
                'std_solving_time': float(np.std(solve_times)),
                'avg_total_time': float(np.mean(total_times)),
                'std_total_time': float(np.std(total_times)),
                'min_solving_time': float(np.min(solve_times)),
                'max_solving_time': float(np.max(solve_times)),
                'successful_trials': len(T_results)
            }
    
    # Save experiment results
    experiment_data = {
        'experiment_type': 'tlp_solving',
        'parameters': {
            'T_values': T_values,
            'num_trials': num_trials,
            'message_length': len(message),
            'message': message.decode()
        },
        'results': all_results,
        'statistics': stats_by_T,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    result_filename = f"tlp_solving_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_filename}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for T in T_values:
        if str(T) in stats_by_T:
            stats = stats_by_T[str(T)]
            print(f"\nT = {T}:")
            print(f"  Avg Generation Time: {stats['avg_generation_time']:.4f} ± {stats['std_generation_time']:.4f} s")
            print(f"  Avg Solving Time:    {stats['avg_solving_time']:.4f} ± {stats['std_solving_time']:.4f} s")
            print(f"  Total Time:          {stats['avg_total_time']:.4f} ± {stats['std_total_time']:.4f} s")
            print(f"  Successful Trials:   {stats['successful_trials']}/{num_trials}")
    print("="*70)
    
    return experiment_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time-Lock Puzzle Solving Experiment")
    parser.add_argument('--T-values', type=int, nargs='+', default=[4500 * 2, 4500 * 15, 4500 * 40, 4500 * 120],
                        help='T values to test (default: 256 1024 4096 16384)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials per T value (default: 5)')
    parser.add_argument('--message', type=str, default="This is a secret message!",
                        help='Message to encrypt (default: "This is a secret message!")')
    parser.add_argument('--output-dir', type=str, default='./experiments/tlp_solving',
                        help='Output directory (default: ./experiments/tlp_solving)')
    
    args = parser.parse_args()
    
    print("\nStarting TLP Solving Experiment...\n")
    
    results = run_tlp_solving_experiment(
        T_values=args.T_values,
        num_trials=args.trials,
        message=args.message.encode(),
        output_dir=args.output_dir
    )
    
    print("\n✓ Experiment complete!")
