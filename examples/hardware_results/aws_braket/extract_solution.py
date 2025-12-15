"""
Extract solution probabilities from HHL measurement results.
For 4×4 problems, we need to extract the 2 solution qubits.
"""

import json
import numpy as np
import os

# Load results - find the results file
script_dir = os.path.dirname(os.path.realpath(__file__))
result_file = os.path.join(script_dir, 'braket_results_custom_N4_matrix_hhl0.json')

with open(result_file, 'r') as f:
    data = json.load(f)

counts = data['enhanced_results'][0]['counts']
total_shots = data['enhanced_results'][0]['shots']

print("="*60)
print("HHL Solution Extraction for 4×4 Problem")
print("="*60)

# Circuit structure: [ancilla(1)][clock(5)][solution(2)] = 8 qubits
# We want to extract the solution qubits (rightmost 2 bits)

# Group by ancilla and solution qubits
solution_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
successful_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
failed_counts = {'00': 0, '01': 0, '10': 0, '11': 0}

for bitstring, count in counts.items():
    # Parse bitstring
    ancilla = bitstring[0]  # Leftmost bit
    clock = bitstring[1:6]   # Middle 5 bits
    solution = bitstring[6:8]  # Rightmost 2 bits
    
    # Count all measurements
    solution_counts[solution] += count
    
    # Separate by ancilla
    if ancilla == '1':
        successful_counts[solution] += count
    else:
        failed_counts[solution] += count

# Calculate probabilities
print("\n--- All Measurements (Ancilla 0 or 1) ---")
total_all = sum(solution_counts.values())
for sol in ['00', '01', '10', '11']:
    prob = solution_counts[sol] / total_all
    print(f"Solution |{sol}⟩: {solution_counts[sol]:4d} / {total_all} = {prob:.4f}")

print("\n--- Successful Measurements (Ancilla = 1) ---")
total_success = sum(successful_counts.values())
success_rate = total_success / total_shots
print(f"Success rate: {total_success}/{total_shots} = {success_rate:.2%}")
print()
for sol in ['00', '01', '10', '11']:
    prob = successful_counts[sol] / total_success if total_success > 0 else 0
    print(f"Solution |{sol}⟩: {successful_counts[sol]:4d} / {total_success} = {prob:.4f}")

print("\n--- Failed Measurements (Ancilla = 0) ---")
total_failed = sum(failed_counts.values())
for sol in ['00', '01', '10', '11']:
    prob = failed_counts[sol] / total_failed if total_failed > 0 else 0
    print(f"Solution |{sol}⟩: {failed_counts[sol]:4d} / {total_failed} = {prob:.4f}")

# Normalized solution probabilities (post-selected on ancilla=1)
print("\n" + "="*60)
print("FINAL SOLUTION PROBABILITIES (Ancilla = 1)")
print("="*60)
solution_probs = []
for sol in ['00', '01', '10', '11']:
    prob = successful_counts[sol] / total_success if total_success > 0 else 0
    solution_probs.append(prob)
    binary = int(sol, 2)
    print(f"|{sol}⟩ (state {binary}): {prob:.4f}")

# Compute amplitudes (square root of probabilities)
print("\n--- Solution Amplitudes ---")
amplitudes = np.sqrt(solution_probs)
for i, amp in enumerate(amplitudes):
    print(f"Amplitude for state {i}: {amp:.4f}")

# Normalize amplitudes
norm = np.linalg.norm(amplitudes)
normalized_amplitudes = amplitudes / norm if norm > 0 else amplitudes
print("\n--- Normalized Solution Vector ---")
print("x =", normalized_amplitudes)

# Compare with classical solution
A = np.array(data['problem_list'][0]['A_matrix'])
b = np.array(data['problem_list'][0]['b_vector'])
x_classical = np.linalg.solve(A, b)
x_classical_normalized = x_classical / np.linalg.norm(x_classical)

print("\n--- Classical Solution (Normalized) ---")
print("x_classical =", x_classical_normalized)

print("\n--- Comparison ---")
fidelity = np.abs(np.dot(normalized_amplitudes, x_classical_normalized))**2
print(f"Fidelity: {fidelity:.4f}")
print(f"L2 Error: {np.linalg.norm(normalized_amplitudes - x_classical_normalized):.4f}")

print("\n" + "="*60)
