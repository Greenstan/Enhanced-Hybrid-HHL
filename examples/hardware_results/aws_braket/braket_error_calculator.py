"""
AWS Braket HHL Error Calculator

This script analyzes AWS Braket HHL results by:
1. Loading result files
2. Computing classical solutions
3. Extracting quantum solutions from measurement results
4. Calculating fidelity and error metrics
5. Generating comparison statistics
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path to import enhanced_hybrid_hhl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from enhanced_hybrid_hhl import (
    QuantumLinearSystemProblem,
    QuantumLinearSystemSolver,
    ExampleQLSP
)

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))


def load_results(filename: str) -> Dict:
    """Load results from JSON file."""
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'r') as file:
        return json.load(file)


def compute_classical_solution(A_matrix: List[List[float]], 
                               b_vector: List[float] = None) -> np.ndarray:
    """
    Compute classical solution to the linear system Ax = b.
    
    Args:
        A_matrix: The coefficient matrix
        b_vector: The right-hand side vector (default: [1, 0])
        
    Returns:
        Normalized solution vector
    """
    A = np.array(A_matrix)
    
    # Default b vector if not provided
    if b_vector is None:
        b = np.array([1.0, 0.0])
    else:
        b = np.array(b_vector)
    
    # Solve the system
    x = np.linalg.solve(A, b)
    
    # Normalize
    x_normalized = x / np.linalg.norm(x)
    
    return x_normalized


def extract_quantum_solution_swap_test(counts: Dict[str, int]) -> float:
    """
    Extract fidelity from swap test results.
    
    The swap test measures the overlap between quantum and ideal states.
    Ancilla in |0⟩ indicates successful state preparation.
    
    Args:
        counts: Measurement counts from the quantum circuit
        
    Returns:
        Fidelity estimate
    """
    # Parse counts for swap test results
    # Format is typically 'ancilla other_qubits'
    counts_01 = 0  # Ancilla = 0, success qubit = 1
    counts_11 = 0  # Ancilla = 1, success qubit = 1
    
    for bitstring, count in counts.items():
        # Remove spaces from bitstring
        bits = bitstring.replace(' ', '')
        
        # Check ancilla (first bit) and success flag
        if len(bits) >= 2:
            ancilla = bits[0]
            success = bits[-1] if len(bits) > 1 else bits[0]
            
            if ancilla == '0' and success == '1':
                counts_01 += count
            elif ancilla == '1' and success == '1':
                counts_11 += count
    
    # Calculate fidelity from swap test
    if counts_01 + counts_11 == 0:
        return 0.0
    
    if counts_01 <= counts_11:
        return 0.0
    
    prob_0 = counts_01 / (counts_01 + counts_11)
    fidelity = np.sqrt(2 * prob_0 - 1) if prob_0 >= 0.5 else 0.0
    
    return fidelity


def extract_quantum_amplitudes(counts: Dict[str, int], 
                               num_solution_qubits: int = 2) -> np.ndarray:
    """
    Extract quantum state amplitudes from measurement counts.
    
    For HHL, we measure the solution register to extract amplitudes.
    This assumes the ancilla qubit showed success (|1⟩).
    
    Args:
        counts: Measurement counts
        num_solution_qubits: Number of qubits in the solution register
        
    Returns:
        Estimated amplitude vector
    """
    # Initialize amplitude array
    amplitudes = np.zeros(2**num_solution_qubits)
    total_counts = 0
    
    for bitstring, count in counts.items():
        # Parse bitstring (format varies by backend)
        bits = bitstring.replace(' ', '')
        
        # Extract solution register bits (assuming they're in specific positions)
        # This depends on the circuit structure
        # For standard HHL: |ancilla⟩|clock⟩|solution⟩
        
        # If ancilla (first bit) is 1, this is a successful measurement
        if len(bits) > 0 and bits[0] == '1':
            # Extract solution bits (last num_solution_qubits bits)
            solution_bits = bits[-num_solution_qubits:]
            solution_idx = int(solution_bits, 2)
            
            amplitudes[solution_idx] += count
            total_counts += count
    
    # Normalize to get probabilities, then take sqrt for amplitudes
    if total_counts > 0:
        probabilities = amplitudes / total_counts
        amplitudes = np.sqrt(probabilities)
        
        # Normalize amplitude vector
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
    
    return amplitudes


def calculate_fidelity(classical_solution: np.ndarray, 
                      quantum_solution: np.ndarray) -> float:
    """
    Calculate fidelity between classical and quantum solutions.
    
    Fidelity = |⟨ψ_classical|ψ_quantum⟩|²
    
    Args:
        classical_solution: Normalized classical solution
        quantum_solution: Normalized quantum solution
        
    Returns:
        Fidelity value between 0 and 1
    """
    # Ensure both are normalized
    classical_norm = classical_solution / np.linalg.norm(classical_solution)
    quantum_norm = quantum_solution / np.linalg.norm(quantum_solution)
    
    # Calculate inner product
    overlap = np.abs(np.dot(classical_norm, quantum_norm))
    
    # Fidelity is the square of the overlap
    fidelity = overlap ** 2
    
    return fidelity


def calculate_error_metrics(classical_solution: np.ndarray,
                           quantum_solution: np.ndarray) -> Dict:
    """
    Calculate various error metrics between solutions.
    
    Args:
        classical_solution: Classical solution
        quantum_solution: Quantum solution
        
    Returns:
        Dictionary of error metrics
    """
    fidelity = calculate_fidelity(classical_solution, quantum_solution)
    
    # L2 (Euclidean) error
    l2_error = np.linalg.norm(classical_solution - quantum_solution)
    
    # Relative L2 error
    relative_l2_error = l2_error / np.linalg.norm(classical_solution)
    
    # State preparation error (from fidelity)
    state_prep_error = np.sqrt(2 * (1 - fidelity))
    
    # Cosine similarity
    cosine_sim = np.dot(classical_solution, quantum_solution) / (
        np.linalg.norm(classical_solution) * np.linalg.norm(quantum_solution)
    )
    
    return {
        'fidelity': fidelity,
        'l2_error': l2_error,
        'relative_l2_error': relative_l2_error,
        'state_prep_error': state_prep_error,
        'cosine_similarity': cosine_sim,
        '1_minus_fidelity': 1 - fidelity
    }


def analyze_results_file(filename: str) -> Dict:
    """
    Analyze a single results file and compute error metrics.
    
    Args:
        filename: Name of the results JSON file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print('='*60)
    
    # Load results
    data = load_results(filename)
    
    # Extract data
    problem_list = data.get('problem_list', [])
    shots = data.get('shots', 1000)
    backend = data.get('backend', 'Unknown')
    
    canonical_results = data.get('canonical_results', [])
    canonical_depths = data.get('canonical_depths', [])
    
    hybrid_results = data.get('hybrid_results', [])
    hybrid_depths = data.get('hybrid_depths', [])
    
    enhanced_results = data.get('enhanced_results', [])
    enhanced_depths = data.get('enhanced_depths', [])
    
    print(f"\nBackend: {backend}")
    print(f"Shots: {shots}")
    print(f"Number of problems: {len(problem_list)}")
    
    # Initialize result storage
    canonical_metrics = []
    hybrid_metrics = []
    enhanced_metrics = []
    
    # Analyze each problem
    for i, A_matrix in enumerate(problem_list):
        print(f"\n--- Problem {i+1}/{len(problem_list)} ---")
        
        # Compute classical solution
        classical_sol = compute_classical_solution(A_matrix)
        print(f"Classical solution: {classical_sol}")
        
        # Analyze canonical results
        if i < len(canonical_results):
            result = canonical_results[i]
            if result.get('status') == 'COMPLETED':
                counts = result.get('counts', {})
                fidelity = extract_quantum_solution_swap_test(counts)
                
                # Use fidelity to estimate quantum solution
                # This is an approximation
                error = np.sqrt(2 * (1 - fidelity)) if fidelity < 1 else 0
                
                canonical_metrics.append({
                    'fidelity': fidelity,
                    'state_prep_error': error,
                    'depth': canonical_depths[i] if i < len(canonical_depths) else None,
                    'counts': counts
                })
                print(f"  Canonical: fidelity={fidelity:.4f}, error={error:.4f}")
        
        # Analyze hybrid results
        if i < len(hybrid_results):
            result = hybrid_results[i]
            if result.get('status') == 'COMPLETED':
                counts = result.get('counts', {})
                fidelity = extract_quantum_solution_swap_test(counts)
                error = np.sqrt(2 * (1 - fidelity)) if fidelity < 1 else 0
                
                hybrid_metrics.append({
                    'fidelity': fidelity,
                    'state_prep_error': error,
                    'depth': hybrid_depths[i] if i < len(hybrid_depths) else None,
                    'counts': counts
                })
                print(f"  Hybrid: fidelity={fidelity:.4f}, error={error:.4f}")
        
        # Analyze enhanced results
        if i < len(enhanced_results):
            result = enhanced_results[i]
            if result.get('status') == 'COMPLETED':
                counts = result.get('counts', {})
                fidelity = extract_quantum_solution_swap_test(counts)
                error = np.sqrt(2 * (1 - fidelity)) if fidelity < 1 else 0
                
                enhanced_metrics.append({
                    'fidelity': fidelity,
                    'state_prep_error': error,
                    'depth': enhanced_depths[i] if i < len(enhanced_depths) else None,
                    'counts': counts
                })
                print(f"  Enhanced: fidelity={fidelity:.4f}, error={error:.4f}")
    
    # Compute averages
    analysis = {
        'filename': filename,
        'backend': backend,
        'shots': shots,
        'num_problems': len(problem_list),
        'canonical': compute_average_metrics(canonical_metrics),
        'hybrid': compute_average_metrics(hybrid_metrics),
        'enhanced': compute_average_metrics(enhanced_metrics)
    }
    
    return analysis


def compute_average_metrics(metrics_list: List[Dict]) -> Dict:
    """Compute average metrics from a list of individual metrics."""
    if not metrics_list:
        return {
            'count': 0,
            'avg_fidelity': 0,
            'avg_error': 0,
            'avg_depth': 0,
            'std_fidelity': 0,
            'std_error': 0
        }
    
    fidelities = [m['fidelity'] for m in metrics_list]
    errors = [m['state_prep_error'] for m in metrics_list]
    depths = [m['depth'] for m in metrics_list if m['depth'] is not None]
    
    return {
        'count': len(metrics_list),
        'avg_fidelity': np.mean(fidelities),
        'avg_error': np.mean(errors),
        'avg_depth': np.mean(depths) if depths else 0,
        'std_fidelity': np.std(fidelities),
        'std_error': np.std(errors),
        'min_fidelity': np.min(fidelities),
        'max_fidelity': np.max(fidelities),
        'min_error': np.min(errors),
        'max_error': np.max(errors)
    }


def print_summary(analysis: Dict):
    """Print summary of analysis results."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nFile: {analysis['filename']}")
    print(f"Backend: {analysis['backend']}")
    print(f"Problems analyzed: {analysis['num_problems']}")
    
    for method in ['canonical', 'hybrid', 'enhanced']:
        metrics = analysis[method]
        if metrics['count'] > 0:
            print(f"\n{method.upper()} HHL:")
            print(f"  Sample size: {metrics['count']}")
            print(f"  Avg Fidelity: {metrics['avg_fidelity']:.4f} ± {metrics['std_fidelity']:.4f}")
            print(f"  Avg Error: {metrics['avg_error']:.4f} ± {metrics['std_error']:.4f}")
            print(f"  Avg Depth: {metrics['avg_depth']:.1f}")
            print(f"  Fidelity range: [{metrics['min_fidelity']:.4f}, {metrics['max_fidelity']:.4f}]")
            print(f"  Error range: [{metrics['min_error']:.4f}, {metrics['max_error']:.4f}]")


def save_analysis(analysis: Dict, output_filename: str):
    """Save analysis results to JSON file."""
    output_path = os.path.join(script_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✓ Analysis saved to: {output_filename}")


def main():
    """Main function to analyze all result files."""
    print("="*60)
    print("AWS Braket HHL Error Calculator")
    print("="*60)
    
    # Find all result files
    result_files = [f for f in os.listdir(script_dir)
                   if f.startswith('braket_results_N2_matrix_hhl')
                   and f.endswith('.json')]
    
    if not result_files:
        print("\nNo result files found!")
        print(f"Looking in: {script_dir}")
        return
    
    print(f"\nFound {len(result_files)} result file(s)")
    
    # Analyze each file
    all_analyses = []
    
    for result_file in sorted(result_files):
        analysis = analyze_results_file(result_file)
        print_summary(analysis)
        
        # Save individual analysis
        analysis_filename = result_file.replace('braket_results', 'braket_analysis')
        save_analysis(analysis, analysis_filename)
        
        all_analyses.append(analysis)
    
    # Compute overall statistics
    if len(all_analyses) > 1:
        print("\n" + "="*60)
        print("OVERALL STATISTICS (All Files)")
        print("="*60)
        
        for method in ['canonical', 'hybrid', 'enhanced']:
            all_fidelities = []
            all_errors = []
            
            for analysis in all_analyses:
                metrics = analysis[method]
                if metrics['count'] > 0:
                    all_fidelities.append(metrics['avg_fidelity'])
                    all_errors.append(metrics['avg_error'])
            
            if all_fidelities:
                print(f"\n{method.upper()} HHL (across all files):")
                print(f"  Avg Fidelity: {np.mean(all_fidelities):.4f} ± {np.std(all_fidelities):.4f}")
                print(f"  Avg Error: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
