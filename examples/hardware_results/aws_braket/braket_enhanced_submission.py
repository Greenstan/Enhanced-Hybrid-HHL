"""
AWS Braket Enhanced HHL Submission Script (N=4 Problems)

Runs only Enhanced Hybrid HHL on 4×4 linear system problems.
Designed for real problem matrices (not just preprocessing data).
"""

import sys
import os
import json
import numpy as np

from enhanced_hybrid_hhl import (HHL, 
                                 Lee_preprocessing,  
                                 QuantumLinearSystemProblem, 
                                 QuantumLinearSystemSolver,
                                 EnhancedHybridInversion,
                                 HHL_Result,
                                 ideal_preprocessing,
                                 list_preprocessing)

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_braket_provider import BraketProvider

import boto3

# Configuration
AWS_REGION = "eu-west-2"
DEVICE_ARN = "SV1"
S3_BUCKET = None
S3_PREFIX = "hhl-n4-results"

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load problem file - look for N4 problem files
problem_files = [
    "custom_4x4_problem.json",
    # 'torino_N4_matrix_hhl.json',
    # '../torino_N4_matrix_hhl.json',
    # 'example_problems_N4.json',
    # '../problem_generators/example_problems_N4.json'
]

problem_file = None
for pf in problem_files:
    test_path = os.path.join(script_dir, pf)
    if os.path.exists(test_path):
        problem_file = test_path
        break

if problem_file is None:
    print("❌ No N4 problem file found!")
    print(f"Searched for: {problem_files}")
    sys.exit(1)

print(f"Loading problems from: {os.path.basename(problem_file)}")
with open(problem_file, 'r') as file:
    json_data = json.load(file)

# Setup AWS Braket backend
boto_session = boto3.Session(region_name=AWS_REGION)
provider = BraketProvider()
backend = provider.get_backend(DEVICE_ARN)

if S3_BUCKET:
    backend.set_options(
        s3_destination_folder=(S3_BUCKET, S3_PREFIX)
    )

# print(f"✓ Using AWS Braket Backend: {backend.name()}")
print(f"  Region: {AWS_REGION}")

# Parse problem list
if 'problem_list' in json_data:
    problem_data = json_data['problem_list']
else:
    problem_data = json_data

# Convert to QuantumLinearSystemProblem objects
problem_list = []
for p in problem_data:
    if isinstance(p, dict) and 'A_matrix' in p:
        # Format: {"A_matrix": [...], "b_vector": [...]}
        A_matrix = np.array(p['A_matrix'])
        b_vector = np.array(p['b_vector']).flatten()
        problem_list.append(QuantumLinearSystemProblem(A_matrix, b_vector))
    else:
        print(f"⚠ Warning: Skipping invalid problem format: {p}")

print(f"✓ Loaded {len(problem_list)} problems")
print(f"  Problem size: {len(problem_list[0].A_matrix)}×{len(problem_list[0].A_matrix)}")

# Configuration
k_qubits = 7
probability_threshold = 0

# Initialize result storage
used_problem_list = []
ideal_preprocessing_list = []

enhanced_preprocessing_list = []
enhanced_preprocessing_depth_list = []
enhanced_ids = []
enhanced_depths = []
enhanced_results = []


def ensure_all_qubits_measured(circuit):
    """Ensure all qubits are measured for AWS Braket compatibility."""
    num_qubits = circuit.num_qubits
    
    measured_qubits = set()
    measured_mapping = {}
    
    for instr in circuit.data:
        if instr.operation.name == 'measure':
            qubit_idx = circuit.qubits.index(instr.qubits[0])
            clbit_idx = circuit.clbits.index(instr.clbits[0])
            measured_qubits.add(qubit_idx)
            measured_mapping[qubit_idx] = clbit_idx
    
    unmeasured = sorted([i for i in range(num_qubits) if i not in measured_qubits])
    
    if not unmeasured:
        return circuit
    
    print(f"    AWS Braket: Adding measurements for {len(unmeasured)} unmeasured qubits")
    
    if circuit.num_clbits < num_qubits:
        additional_bits = num_qubits - circuit.num_clbits
        aux_creg = ClassicalRegister(additional_bits, f'braket_aux')
        circuit.add_register(aux_creg)
    
    used_clbits = set(measured_mapping.values())
    
    for qubit_idx in unmeasured:
        for clbit_idx in range(circuit.num_clbits):
            if clbit_idx not in used_clbits:
                circuit.measure(qubit_idx, clbit_idx)
                used_clbits.add(clbit_idx)
                break
    
    # Verify
    measured_qubits_final = set()
    for instr in circuit.data:
        if instr.operation.name == 'measure':
            measured_qubits_final.add(circuit.qubits.index(instr.qubits[0]))
    
    if len(measured_qubits_final) != num_qubits:
        raise ValueError(
            f"AWS Braket validation failed: Only {len(measured_qubits_final)} out of "
            f"{num_qubits} qubits are measured."
        )
    
    print(f"    ✓ All {num_qubits} qubits are now measured")
    return circuit


def get_braket_result(circuit: QuantumCircuit,
                      problem: QuantumLinearSystemProblem,
                      shots: int = 3000):
    """Execute HHL circuit on AWS Braket."""
    hhl_circ = circuit.copy()
    
    print(f"    Original circuit: {hhl_circ.num_qubits} qubits, {hhl_circ.num_clbits} clbits")
    
    # Ensure all qubits measured before transpilation
    hhl_circ = ensure_all_qubits_measured(hhl_circ)
    
    # Transpile
    # print(f"    Transpiling for {backend.name()}...")
    transpiled_circuit = transpile(hhl_circ, backend=backend, optimization_level=1)
    
    print(f"    Transpiled: {transpiled_circuit.num_qubits} qubits, "
          f"{transpiled_circuit.num_clbits} clbits, depth {transpiled_circuit.depth()}")
    
    # Ensure all qubits still measured after transpilation
    transpiled_circuit = ensure_all_qubits_measured(transpiled_circuit)
    
    # Create result object
    result = HHL_Result()
    result.depth = transpiled_circuit.depth()
    result.circuit_depth = transpiled_circuit.depth()
    
    # Submit to AWS Braket
    print(f"    Submitting to AWS Braket (shots={shots})...")
    try:
        job = backend.run(transpiled_circuit, shots=shots)
        result.circuit_results = job
        result.task_id = job.job_id()
        print(f"    ✓ Task submitted: {result.task_id}")
    except Exception as e:
        print(f"    ✗ Submission failed: {e}")
        measured_count = sum(1 for instr in transpiled_circuit.data 
                           if instr.operation.name == 'measure')
        print(f"    Debug: Circuit has {transpiled_circuit.num_qubits} qubits, "
              f"{measured_count} measurements")
        raise
    
    return result


# Main execution
print("\n" + "="*60)
print("AWS Braket Enhanced HHL Submission (N=4)")
print("="*60 + "\n")

iteration = 0  # Can be changed for multiple runs

print(f"\n--- Processing {len(problem_list)} problems ---")

for i, problem in enumerate(problem_list):
    print(f"\n[{i+1}/{len(problem_list)}] Problem {i+1}")
    
    # Store problem (both matrix and vector)
    used_problem_list.append({
        'A_matrix': problem.A_matrix.tolist(),
        'b_vector': problem.b_vector.tolist()
    })
    
    # Compute ideal preprocessing
    solution = QuantumLinearSystemSolver(problem)
    ideal_x = solution.ideal_x_statevector
    ideal_preprocessing_list.append(ideal_preprocessing(problem))
    
    # Enhanced Hybrid HHL
    print("  → Running Enhanced Hybrid HHL...")
    
    # Use ideal preprocessing for simplicity (or load from preprocessing file)
    e_preprocessing = ideal_preprocessing
    
    Enhanced_H_HHL = HHL(
        preprocessing=e_preprocessing,
        eigenvalue_inversion=EnhancedHybridInversion
    )
    
    enhanced_result = Enhanced_H_HHL.estimate(
        problem=problem,
        num_clock_qubits=k_qubits,
        max_eigenvalue=10,
        quantum_conditional_logic=False,
        probability_threshold=probability_threshold,
        get_result_function=get_braket_result
    )
    
    enhanced_ids.append(enhanced_result.task_id)
    enhanced_depths.append(enhanced_result.depth)
    print(f"     Task ID: {enhanced_result.task_id}")
    print(f"     Circuit depth: {enhanced_result.depth}")

# Save results
data = {
    'problem_list': used_problem_list,
    'shots': 3000,
    'backend': "sv1",
    'device_arn': DEVICE_ARN,
    'aws_region': AWS_REGION,
    'problem_source': os.path.basename(problem_file),
    'ideal_preprocessing_list': ideal_preprocessing_list,
    'probability_threshold': probability_threshold,
    'k_qubits': k_qubits,
    
    'enhanced_ids': enhanced_ids,
    'enhanced_depths': enhanced_depths,
    'enhanced_results': []  # Will be filled by retrieval script
}

# Save to file
file_name = f'braket_enhanced_N4_matrix_hhl{iteration}.json'
file_path = os.path.join(script_dir, file_name)

with open(file_path, "w") as json_file:
    json.dump(data, json_file, indent=2)

print(f"\n✓ Results saved to: {file_name}")

print("\n" + "="*60)
print("AWS Braket Enhanced HHL Submission Complete!")
print("="*60)
print(f"\n{len(enhanced_ids)} tasks submitted")
print("\nUse braket_result_retrieval.py to retrieve results later.")
