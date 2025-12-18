"""
AWS Braket Projection Result Retrieval and Analysis Script

Retrieves results from projection-based HHL measurements and reconstructs
the solution vector from the projection measurement outcomes.
"""

import datetime
import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple

from qiskit_braket_provider import BraketProvider
import boto3

# Configuration
AWS_REGION = "eu-west-2"
DEVICE_ARN = "SV1"

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))

# Find the most recent projection results file
result_files = [f for f in os.listdir(script_dir) 
                if f.startswith('braket_enhanced_projection_N4_matrix_hhl_20251217_152356') and f.endswith('.json')]

if not result_files:
    print("❌ No projection result files found!")
    sys.exit(1)

# Use most recent file
result_file = sorted(result_files)[-1]
result_path = os.path.join(script_dir, result_file)

print(f"Loading projection results from: {result_file}")
with open(result_path, 'r') as file:
    data = json.load(file)

# Setup AWS Braket backend
boto_session = boto3.Session(region_name=AWS_REGION)
provider = BraketProvider()
backend = provider.get_backend(DEVICE_ARN)

print(f"✓ Using AWS Braket Backend: {DEVICE_ARN}")
print(f"  Region: {AWS_REGION}")


def extract_solution_probability_from_counts(counts: dict, 
                                             target_basis_state: int,
                                             num_solution_qubits: int,
                                             debug: bool = False) -> Tuple[float, float]:
    """
    Extract the probability of a specific basis state from measurement counts.
    
    Args:
        counts: Measurement counts dictionary from Braket
        target_basis_state: The basis state index we're measuring (0, 1, 2, 3, etc.)
        num_solution_qubits: Number of qubits encoding the solution
        debug: Print debug information about bit extraction
        
    Returns:
        (probability_given_success, overall_success_rate)
    """
    total_counts = sum(counts.values())
    success_counts = 0
    target_and_success_counts = 0
    
    # Convert target basis state to binary string
    target_bits = format(target_basis_state, f'0{num_solution_qubits}b')
    
    if debug:
        print(f"\n  Debug: Looking for target basis state |{target_basis_state}⟩ = |{target_bits}⟩")
        print(f"  Sample bitstrings from counts:")
        for i, (bitstring, count) in enumerate(list(counts.items())[:5]):
            print(f"    {bitstring}: {count}")
    
    for bitstring, count in counts.items():
        # Qiskit/Braket bitstring format: measurements are in reverse qubit order
        # Bitstring bit index i corresponds to qubit (num_qubits - 1 - i)
        # So rightmost bit = highest qubit number
        
        bits = bitstring.replace(' ', '')
        
        # In the HHL circuit:
        # Qubits: [flag(0), clock(1..k), b_reg(k+1..k+n)]
        # We added measurements for solution qubits to a separate classical register
        # Need to find which classical bits correspond to solution qubits
        
        # The flag qubit (qubit 0) is the RIGHTMOST bit in bitstring
        # because bitstrings are reversed: bit[i] = qubit[n-1-i]
        flag_bit = int(bits[-1])  # Rightmost bit = qubit 0 = flag
        
        # Solution qubits are the highest numbered qubits
        # For 10-qubit circuit: qubits 8,9 are solution
        # In bitstring: bits 0,1 (leftmost) correspond to qubits 9,8
        # Bitstring format: qubit n-1 is at index 0, qubit n-2 at index 1, etc.
        # So for 2 qubits: leftmost bit = higher qubit = MSB, which is already in correct order
        solution_bits = bits[:num_solution_qubits]  # Leftmost bits are already in correct order
        
        if debug and success_counts < 3:
            print(f"    Bitstring: {bits}, flag={flag_bit}, solution_bits={solution_bits}")
        
        if flag_bit == 1:
            success_counts += count
            
            # Check if solution matches target
            if solution_bits == target_bits:
                target_and_success_counts += count
    
    # Calculate probabilities
    overall_success_rate = success_counts / total_counts if total_counts > 0 else 0.0
    prob_given_success = target_and_success_counts / success_counts if success_counts > 0 else 0.0
    
    if debug:
        print(f"  Success rate: {overall_success_rate:.4f}")
        print(f"  P(target|success): {prob_given_success:.4f}")
    
    return prob_given_success, overall_success_rate


def reconstruct_solution_vector(projection_results: List[Dict],
                                num_solution_qubits: int) -> Tuple[np.ndarray, Dict]:
    """
    Reconstruct the solution vector from projection measurement results.
    
    Args:
        projection_results: List of projection task results with counts
        num_solution_qubits: Number of qubits in solution register
        
    Returns:
        (solution_probabilities, metadata_dict)
    """
    dimension = 2**num_solution_qubits
    solution_probs = np.zeros(dimension)
    success_rates = []
    
    metadata = {
        'measured_components': [],
        'success_rates': [],
        'raw_probabilities': []
    }
    
    for proj_result in projection_results:
        if 'error' in proj_result or proj_result.get('counts') is None:
            continue
        
        basis_state = proj_result['basis_state']
        counts = proj_result['counts']
        
        # Extract probability for this basis state
        prob, success_rate = extract_solution_probability_from_counts(
            counts, basis_state, num_solution_qubits
        )
        
        solution_probs[basis_state] = prob
        success_rates.append(success_rate)
        
        metadata['measured_components'].append(basis_state)
        metadata['success_rates'].append(success_rate)
        metadata['raw_probabilities'].append(prob)
    
    # Normalize probabilities
    total_prob = np.sum(solution_probs)
    if total_prob > 0:
        solution_probs /= total_prob
    
    metadata['average_success_rate'] = np.mean(success_rates) if success_rates else 0.0
    metadata['total_probability'] = float(total_prob)
    
    return solution_probs, metadata


def calculate_fidelity(measured_probs: np.ndarray, ideal_vector: np.ndarray) -> float:
    """
    Calculate fidelity between measured probabilities and ideal solution.
    
    Args:
        measured_probs: Probability distribution from measurements
        ideal_vector: Ideal solution statevector
        
    Returns:
        Fidelity value (0 to 1)
    """
    # Ideal probabilities
    ideal_probs = np.abs(ideal_vector)**2
    ideal_probs /= np.sum(ideal_probs)
    
    # Classical fidelity for probability distributions
    fidelity = np.sum(np.sqrt(measured_probs * ideal_probs))**2
    
    return fidelity


# Retrieve results
print(f"\n--- Retrieving Results from AWS Braket ---\n")

num_solution_qubits = data['num_solution_qubits']
enhanced_projection_results = data['enhanced_projection_results']

updated_results = []

for prob_idx, prob_result in enumerate(enhanced_projection_results):
    print(f"Problem {prob_idx + 1}/{len(enhanced_projection_results)}")
    
    projection_tasks = prob_result['projection_tasks']
    
    # Convert ideal solution from JSON format back to complex numpy array
    ideal_solution_json = prob_result['ideal_solution']
    if isinstance(ideal_solution_json[0], dict):
        # Format: [{'real': x, 'imag': y}, ...]
        ideal_solution = np.array([
            complex(item['real'], item['imag']) 
            for item in ideal_solution_json
        ])
    else:
        # Legacy format: already a list of numbers
        ideal_solution = np.array(ideal_solution_json)
    
    print(f"  Retrieving {len(projection_tasks)} projection measurements...")
    
    # Retrieve each projection task
    retrieved_tasks = []
    for task in projection_tasks:
        task_id = task.get('task_id')
        basis_state = task.get('basis_state')
        
        if task_id is None:
            print(f"    Basis |{basis_state}⟩: No task ID (submission failed)")
            retrieved_tasks.append(task)
            continue
        
        try:
            job = backend.retrieve_job(task_id)
            status = job.status()
            status_str = str(status.name) if hasattr(status, 'name') else str(status)
            
            task_result = {
                'basis_state': basis_state,
                'task_id': task_id,
                'status': status_str,
                'shots': task['shots']
            }
            
            if status_str in ['COMPLETED', 'DONE', 'JobStatus.COMPLETED']:
                result = job.result()
                counts = result.get_counts()
                task_result['counts'] = counts
                
                # Extract probability for this component (enable debug for first task)
                debug = (len(retrieved_tasks) == 0)
                prob, success_rate = extract_solution_probability_from_counts(
                    counts, basis_state, num_solution_qubits, debug=debug
                )
                task_result['component_probability'] = float(prob)
                task_result['success_rate'] = float(success_rate)
                
                print(f"    Basis |{basis_state}⟩: P={prob:.4f}, Success={success_rate:.4f}")
            else:
                print(f"    Basis |{basis_state}⟩: Status={status_str}")
            
            retrieved_tasks.append(task_result)
            
        except Exception as e:
            print(f"    Basis |{basis_state}⟩: Retrieval failed - {e}")
            task['error'] = str(e)
            retrieved_tasks.append(task)
    
    # Reconstruct solution vector
    solution_probs, metadata = reconstruct_solution_vector(
        retrieved_tasks, num_solution_qubits
    )
    
    # Calculate fidelity with ideal solution
    fidelity = calculate_fidelity(solution_probs, ideal_solution)
    
    print(f"\n  Reconstructed solution probabilities: {solution_probs}")
    print(f"  Ideal solution amplitudes: {ideal_solution}")
    print(f"  Ideal probabilities: {np.abs(ideal_solution)**2 / np.sum(np.abs(ideal_solution)**2)}")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  Average success rate: {metadata['average_success_rate']:.4f}\n")
    
    # Update result with retrieved data
    updated_results.append({
        **prob_result,
        'projection_tasks': retrieved_tasks,
        'reconstructed_solution_probabilities': solution_probs.tolist(),
        'fidelity': float(fidelity),
        'metadata': metadata
    })

# Update data with retrieved results
data['enhanced_projection_results'] = updated_results

# Save updated results
current_datetime =  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = result_file.replace('.json', f'_retrieval.json')
output_path = os.path.join(script_dir, output_file)

with open(output_path, 'w') as file:
    json.dump(data, file, indent=2)

print(f"\n✓ Retrieved results saved to: {output_file}")

# Print summary
print("\n" + "="*60)
print("Projection Measurement Summary")
print("="*60)

for i, result in enumerate(updated_results):
    print(f"\nProblem {i+1}:")
    print(f"  Fidelity: {result['fidelity']:.4f}")
    print(f"  Avg success rate: {result['metadata']['average_success_rate']:.4f}")
    print(f"  Components measured: {result['metadata']['measured_components']}")
    print(f"  Solution: {result['reconstructed_solution_probabilities']}")

print("\n" + "="*60)
