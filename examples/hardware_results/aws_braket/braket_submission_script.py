import sys
import os
import json
import numpy as np

from enhanced_hybrid_hhl import (HHL, 
                                 Lee_preprocessing,  
                                 HybridInversion, 
                                 QuantumLinearSystemProblem, 
                                 QuantumLinearSystemSolver,
                                 EnhancedHybridInversion,
                                 HHL_Result,
                                 ExampleQLSP,
                                 ideal_preprocessing,
                                 CanonicalInversion,
                                 list_preprocessing)

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_braket_provider import BraketProvider

# AWS Braket specific imports
import boto3

# Configuration
AWS_REGION = "eu-west-2"  # Change to your preferred region
DEVICE_ARN = "SV1"  # Options: "SV1" (simulator), "TN1" (tensor network), or actual device ARN
S3_BUCKET = None  # Set to your S3 bucket name, or None to use default
S3_PREFIX = "hhl-results"

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))
preprocessing_file_name = 'simulator_small_matrix_preprocessing.json'

# Load preprocessing data - try multiple locations
file_path = os.path.join(script_dir, preprocessing_file_name)
if not os.path.exists(file_path):
    # Try IQM directory
    iqm_dir = os.path.join(os.path.dirname(script_dir), 'iqm')
    file_path = os.path.join(iqm_dir, preprocessing_file_name)
    if not os.path.exists(file_path):
        # Try parent hardware_results directory
        parent_dir = os.path.dirname(script_dir)
        file_path = os.path.join(parent_dir, preprocessing_file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Could not find {preprocessing_file_name}\n"
                f"Searched in:\n"
                f"  - {script_dir}\n"
                f"  - {iqm_dir}\n"
                f"  - {parent_dir}\n"
                f"Please ensure the preprocessing file exists in one of these locations."
            )

print(f"Loading preprocessing data from: {file_path}")
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Setup AWS Braket backend
boto_session = boto3.Session(region_name=AWS_REGION)
provider = BraketProvider()

# Get backend - can be SV1, TN1, or a specific device
backend = provider.get_backend(DEVICE_ARN)

# Configure S3 destination if specified
if S3_BUCKET:
    backend.set_options(
        s3_destination_folder=(S3_BUCKET, S3_PREFIX)
    )

# print(f"✓ Using AWS Braket Backend: {backend.name()}")
print(f"  Region: {AWS_REGION}")
if S3_BUCKET:
    print(f"  S3 Destination: s3://{S3_BUCKET}/{S3_PREFIX}")

# Load problem data
lam_list = json_data['lam_list']
problem_list = [ExampleQLSP(lam) for lam in lam_list]
fixed_result_list = json_data['fixed']
enhanced_fixed_result_list = json_data['enhanced_fixed']

# Configuration
k_qubits = 3
probability_threshold = 0  # 2**(-k_qubits)

# Initialize result storage
ideal_preprocessing_list = []
used_problem_list = []

canonical_ids = []
canonical_depths = []
canonical_results = []

hybrid_preprocessing_list = []
hybrid_preprocessing_depth_list = []
hybrid_ids = []
hybrid_depths = []
hybrid_results = []

enhanced_preprocessing_list = []
enhanced_preprocessing_depth_list = []
enhanced_ids = []
enhanced_depths = []
enhanced_results = []


def ensure_all_qubits_measured(circuit):
    """
    Ensure all qubits in the circuit are measured for AWS Braket compatibility.
    
    AWS Braket requires ALL qubits to be measured. This function:
    1. Identifies which qubits are already measured
    2. Adds classical registers if needed
    3. Measures all unmeasured qubits
    """
    from qiskit import QuantumCircuit
    
    num_qubits = circuit.num_qubits
    
    # Find which qubits are already measured
    measured_qubits = set()
    measured_mapping = {}  # qubit_idx -> clbit_idx
    
    for instr in circuit.data:
        if instr.operation.name == 'measure':
            qubit_idx = circuit.qubits.index(instr.qubits[0])
            clbit_idx = circuit.clbits.index(instr.clbits[0])
            measured_qubits.add(qubit_idx)
            measured_mapping[qubit_idx] = clbit_idx
    
    # Find unmeasured qubits
    unmeasured = sorted([i for i in range(num_qubits) if i not in measured_qubits])
    
    if not unmeasured:
        return circuit  # All qubits already measured
    
    print(f"    AWS Braket: Adding measurements for {len(unmeasured)} unmeasured qubits: {unmeasured}")
    
    # Ensure we have enough classical bits for ALL qubits
    if circuit.num_clbits < num_qubits:
        additional_bits = num_qubits - circuit.num_clbits
        aux_creg = ClassicalRegister(additional_bits, f'braket_aux')
        circuit.add_register(aux_creg)
        print(f"    Added {additional_bits} classical bits (total: {circuit.num_clbits})")
    
    # Find which classical bits are already used
    used_clbits = set(measured_mapping.values())
    
    # Measure each unmeasured qubit to an unused classical bit
    for qubit_idx in unmeasured:
        # Find first available classical bit
        for clbit_idx in range(circuit.num_clbits):
            if clbit_idx not in used_clbits:
                circuit.measure(qubit_idx, clbit_idx)
                used_clbits.add(clbit_idx)
                break
        else:
            raise RuntimeError(
                f"Could not find available classical bit for qubit {qubit_idx}. "
                f"This should not happen."
            )
    
    # Final verification
    measured_qubits_final = set()
    for instr in circuit.data:
        if instr.operation.name == 'measure':
            measured_qubits_final.add(circuit.qubits.index(instr.qubits[0]))
    
    if len(measured_qubits_final) != num_qubits:
        raise ValueError(
            f"AWS Braket validation failed: Only {len(measured_qubits_final)} out of "
            f"{num_qubits} qubits are measured. AWS Braket requires all qubits to be measured."
        )
    
    print(f"    ✓ All {num_qubits} qubits are now measured")
    return circuit


def get_braket_result(circuit: QuantumCircuit,
                      problem: QuantumLinearSystemProblem,
                      shots: int = 100):
    """
    Execute HHL circuit on AWS Braket and return results.
    
    AWS Braket has strict requirements:
    - ALL qubits must be measured
    - Qubits should be contiguous starting from 0
    
    Args:
        circuit: The HHL quantum circuit
        problem: The quantum linear system problem
        shots: Number of measurement shots
        
    Returns:
        HHL_Result object with circuit results and metadata
    """
    # Work on a copy to avoid modifying the original circuit
    hhl_circ = circuit.copy()
    
    print(f"    Original circuit: {hhl_circ.num_qubits} qubits, {hhl_circ.num_clbits} clbits")
    
    # Ensure all qubits are measured BEFORE transpilation
    hhl_circ = ensure_all_qubits_measured(hhl_circ)
    
    # Transpile for the backend
    # print(f"    Transpiling for {backend.name()}...")
    transpiled_circuit = transpile(hhl_circ, backend=backend, optimization_level=1)
    
    print(f"    Transpiled circuit: {transpiled_circuit.num_qubits} qubits, "
          f"{transpiled_circuit.num_clbits} clbits, depth {transpiled_circuit.depth()}")
    
    # Ensure all qubits are STILL measured after transpilation
    # (transpilation might change the circuit structure)
    transpiled_circuit = ensure_all_qubits_measured(transpiled_circuit)
    
    # Create result object
    result = HHL_Result()
    result.depth = transpiled_circuit.depth()
    result.circuit_depth = transpiled_circuit.depth()
    
    # Submit job to AWS Braket
    print(f"    Submitting to AWS Braket (shots={shots})...")
    try:
        job = backend.run(transpiled_circuit, shots=shots)
        result.circuit_results = job
        result.task_id = job.job_id()
        print(f"    ✓ Task submitted: {result.task_id}")
    except Exception as e:
        print(f"    ✗ Submission failed: {e}")
        # Print circuit details for debugging
        measured_count = sum(1 for instr in transpiled_circuit.data 
                           if instr.operation.name == 'measure')
        print(f"    Debug: Circuit has {transpiled_circuit.num_qubits} qubits, "
              f"{measured_count} measurements")
        raise
    
    return result
    # result.circuit_results = job
    
    # # Store task ID for later retrieval
    # result.task_id = job.job_id()
    
    # return result


# Main execution loop
print("\n" + "="*60)
print("Starting AWS Braket HHL Submission")
print("="*60 + "\n")

for iteration in range(3, 6):
    print(f"\n--- Iteration {iteration} ---")
    
    for i, problem in enumerate(problem_list):
        print(f"\nProcessing problem {i+1}/{len(problem_list)}...")
        
        used_problem_list.append(problem.A_matrix.tolist())
        solution = QuantumLinearSystemSolver(problem)
        ideal_x = solution.ideal_x_statevector
        
        ideal_preprocessing_list.append(ideal_preprocessing(problem))
        
        # 1. Canonical HHL
        print("  → Running Canonical HHL...")
        Canonical_HHL = HHL(eigenvalue_inversion=CanonicalInversion)
        
        canonical_result = Canonical_HHL.estimate(
            problem=problem, 
            num_clock_qubits=k_qubits,
            max_eigenvalue=1,
            quantum_conditional_logic=False,
            get_result_function=get_braket_result
        )
        
        canonical_ids.append(canonical_result.task_id)
        canonical_depths.append(canonical_result.depth)
        print(f"     Task ID: {canonical_result.task_id}")
        
        # 2. Hybrid HHL (Yalovetzky)
        print("  → Running Hybrid HHL...")
        y_preprocessing = list_preprocessing(
            fixed_result_list[i][0], 
            fixed_result_list[i][1]
        )
        Yalovetzky_H_HHL = HHL(
            preprocessing=y_preprocessing,
            eigenvalue_inversion=HybridInversion
        )
        
        hybrid_result = Yalovetzky_H_HHL.estimate(
            problem=problem,
            num_clock_qubits=k_qubits,
            max_eigenvalue=1,
            quantum_conditional_logic=False,
            get_result_function=get_braket_result
        )
        
        hybrid_ids.append(hybrid_result.task_id)
        hybrid_depths.append(hybrid_result.depth)
        print(f"     Task ID: {hybrid_result.task_id}")
        
        # 3. Enhanced Hybrid HHL
        print("  → Running Enhanced Hybrid HHL...")
        e_preprocessing = list_preprocessing(
            enhanced_fixed_result_list[i][0], 
            enhanced_fixed_result_list[i][1]
        )
        Enhanced_H_HHL = HHL(
            preprocessing=e_preprocessing,
            eigenvalue_inversion=EnhancedHybridInversion
        )
        
        enhanced_result = Enhanced_H_HHL.estimate(
            problem=problem,
            num_clock_qubits=k_qubits,
            max_eigenvalue=20,
            quantum_conditional_logic=False,
            probability_threshold=probability_threshold,
            get_result_function=get_braket_result
        )
        
        enhanced_ids.append(enhanced_result.task_id)
        enhanced_depths.append(enhanced_result.depth)
        print(f"     Task ID: {enhanced_result.task_id}")
    
    # Save results to JSON
    data = {
        'problem_list': used_problem_list,
        'shots': 1000,
        'backend': "SV1",
        'device_arn': DEVICE_ARN,
        'aws_region': AWS_REGION,
        'preprocessing_backend': preprocessing_file_name,
        'ideal_preprocessing_list': ideal_preprocessing_list,
        'probability_threshold': probability_threshold,
        
        'canonical_ids': canonical_ids,
        'canonical_depths': canonical_depths,
        'canonical_results': canonical_results,
        
        'hybrid_preprocessing_list': hybrid_preprocessing_list,
        'hybrid_preprocessing_depth': hybrid_preprocessing_depth_list,
        
        'hybrid_ids': hybrid_ids,
        'hybrid_depths': hybrid_depths,
        'hybrid_results': hybrid_results,
        
        'enhanced_preprocessing_list': enhanced_preprocessing_list,
        'enhanced_preprocessing_depth': enhanced_preprocessing_depth_list,
        
        'enhanced_ids': enhanced_ids,
        'enhanced_depths': enhanced_depths,
        'enhanced_results': enhanced_results,
    }
    
    # Save to file
    file_name = f'simulator_to_braket_N2_matrix_hhl{iteration}.json'
    file_path = os.path.join(script_dir, file_name)
    
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    
    print(f"\n✓ Results saved to: {file_name}")

print("\n" + "="*60)
print("AWS Braket Submission Complete!")
print("="*60)
print("\nTask IDs have been saved. Use braket_result_retrieval.py to retrieve results.")
