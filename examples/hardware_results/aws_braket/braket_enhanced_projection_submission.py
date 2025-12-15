"""
AWS Braket Enhanced HHL Projection Submission Script (N=4 Problems)

Runs Enhanced Hybrid HHL with multiple projection operators to extract
solution components. Supports measuring specific subsets of components
or all components individually.
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
                                 Yalovetzky_preprocessing,
                                 list_preprocessing)

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit_braket_provider import BraketProvider

import boto3

# Configuration
AWS_REGION = "eu-west-2"
DEVICE_ARN = "SV1"
S3_BUCKET = None
S3_PREFIX = "hhl-n4-projection-results"

# Preprocessing configuration
# Options: "ideal", "yalovetzky"
# Note: "yalovetzky" will use quantum circuits for eigenvalue estimation (costs more)
PREPROCESSING_MODE = "ideal"

# Projection configuration
# Options: "all_components", "subset", "custom"
PROJECTION_MODE = "all_components"
# For subset mode: which components to measure (e.g., [0, 1] for first 2 out of 4)
PROJECTION_SUBSET = [0, 1]
# For custom mode: provide custom projection operators
CUSTOM_PROJECTIONS = []

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load problem file - look for N4 problem files
problem_files = [
    "custom_4x4_problem.json",
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
print("LOADED")


# Setup AWS Braket backend
boto_session = boto3.Session(region_name=AWS_REGION)
provider = BraketProvider()
backend = provider.get_backend(DEVICE_ARN)

if S3_BUCKET:
    backend.set_options(
        s3_destination_folder=(S3_BUCKET, S3_PREFIX)
    )

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
k_qubits = 8
probability_threshold = 0

# Initialize result storage
used_problem_list = []
ideal_preprocessing_list = []
enhanced_projection_results = []


def create_computational_basis_projectors(num_qubits, indices=None):
    """
    Create computational basis projection operators.
    
    Args:
        num_qubits: Number of qubits in the solution register
        indices: List of basis state indices to project onto (None = all)
        
    Returns:
        List of (index, projector) tuples
    """
    dimension = 2**num_qubits
    
    if indices is None:
        indices = list(range(dimension))
    
    projectors = []
    for idx in indices:
        # Create |i⟩⟨i| projector
        proj_matrix = np.zeros((dimension, dimension), dtype=complex)
        proj_matrix[idx, idx] = 1.0
        projectors.append((idx, Operator(proj_matrix)))
    
    return projectors


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
    
    return circuit


def get_braket_projection_result(circuit: QuantumCircuit,
                                  problem: QuantumLinearSystemProblem,
                                  projectors: list,
                                  shots: int = 6000):
    """
    Execute HHL circuit on AWS Braket with multiple projection operators.
    
    For each projector, submits the circuit and measures to estimate
    the probability of that basis state in the solution.
    
    Args:
        circuit: HHL quantum circuit (without final measurements on solution qubits)
        problem: QuantumLinearSystemProblem instance
        projectors: List of (index, projector_operator) tuples
        shots: Number of shots per projector measurement
        
    Returns:
        HHL_Result object with task_ids for all projection measurements
    """
    num_solution_qubits = int(np.log2(len(problem.b_vector)))
    
    # Circuit structure: flag[0], c_reg[0], clock_reg[k], b_reg[n]
    # Solution qubits are the last num_solution_qubits (b_reg)
    solution_qubit_start = circuit.num_qubits - num_solution_qubits
    
    projection_tasks = []
    
    print(f"    → Submitting {len(projectors)} projection measurements...")
    
    for proj_idx, (basis_state, projector) in enumerate(projectors):
        # Create a copy of the circuit for this projection
        proj_circuit = circuit.copy()
        
        # The projector measures in computational basis
        # For computational basis, we just need to measure the solution qubits
        # and filter results where we got the target basis state
        
        # Add measurements for solution qubits
        solution_creg = ClassicalRegister(num_solution_qubits, f'sol_meas')
        proj_circuit.add_register(solution_creg)
        
        for i in range(num_solution_qubits):
            proj_circuit.measure(solution_qubit_start + i, solution_creg[i])
        
        # Ensure all qubits measured for AWS Braket
        proj_circuit = ensure_all_qubits_measured(proj_circuit)
        
        # Remove global phase for AWS Braket compatibility
        proj_circuit.global_phase = 0
        
        # Transpile
        transpiled_circuit = transpile(proj_circuit, backend=backend, optimization_level=1)
        transpiled_circuit.global_phase = 0  # Remove any phase added during transpilation
        transpiled_circuit = ensure_all_qubits_measured(transpiled_circuit)
        
        # Submit to AWS Braket
        try:
            job = backend.run(transpiled_circuit, shots=shots)
            task_id = job.job_id()
            
            projection_tasks.append({
                'basis_state': int(basis_state),
                'task_id': task_id,
                'shots': shots,
                'projector_index': proj_idx
            })
            
            print(f"      [{proj_idx+1}/{len(projectors)}] Basis state |{basis_state}⟩: {task_id}")
            
        except Exception as e:
            print(f"      ✗ Projection {proj_idx} failed: {e}")
            projection_tasks.append({
                'basis_state': int(basis_state),
                'task_id': None,
                'error': str(e),
                'projector_index': proj_idx
            })
    
    # Create result object
    result = HHL_Result()
    result.circuit_results = projection_tasks
    result.task_id = projection_tasks[0]['task_id'] if projection_tasks else None
    result.depth = transpiled_circuit.depth() if 'transpiled_circuit' in locals() else None
    result.circuit_depth = result.depth
    result.num_solution_qubits = num_solution_qubits
    result.projection_mode = True
    
    return result


# Determine which projectors to use
def get_projectors_for_mode(num_solution_qubits, mode, subset=None, custom=None):
    """Get projection operators based on mode."""
    if mode == "all_components":
        # Measure all basis states
        print(f"  Projection mode: Measuring all {2**num_solution_qubits} components")
        return create_computational_basis_projectors(num_solution_qubits)
    
    elif mode == "subset":
        # Measure only specified components
        if subset is None:
            subset = [0, 1]
        print(f"  Projection mode: Measuring subset {subset}")
        return create_computational_basis_projectors(num_solution_qubits, indices=subset)
    
    elif mode == "custom":
        # Use custom projectors
        if custom is None or len(custom) == 0:
            print("  Warning: Custom mode but no projectors provided, using all components")
            return create_computational_basis_projectors(num_solution_qubits)
        print(f"  Projection mode: Using {len(custom)} custom projectors")
        return custom
    
    else:
        raise ValueError(f"Unknown projection mode: {mode}")


# Main execution
print("\n" + "="*60)
print("AWS Braket Enhanced HHL Projection Submission (N=4)")
print("="*60 + "\n")

iteration = 0  # Can be changed for multiple runs

# Determine number of solution qubits from first problem
num_solution_qubits = int(np.log2(len(problem_list[0].b_vector)))
print(f"Solution vector size: {len(problem_list[0].b_vector)} ({num_solution_qubits} qubits)")

# Get projectors based on mode
projectors = get_projectors_for_mode(
    num_solution_qubits, 
    PROJECTION_MODE, 
    subset=PROJECTION_SUBSET,
    custom=CUSTOM_PROJECTIONS
)

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
    
    # Enhanced Hybrid HHL with Projection Measurements
    print("  → Running Enhanced Hybrid HHL with projection operators...")
    
    # Select preprocessing method
    if PREPROCESSING_MODE == "yalovetzky":
        print("     Using Yalovetzky preprocessing (quantum-based eigenvalue estimation)")
        y_preprocessing = Yalovetzky_preprocessing(
            clock=k_qubits,
            backend=backend,
            alpha=50,
            max_eigenvalue=10,
            min_prob=2**(-k_qubits)
        )
        # Run preprocessing to get eigenvalues (this submits quantum circuits)
        eigenvalue_list, eigenbasis_projection_list = y_preprocessing.estimate(problem)
        e_preprocessing = list_preprocessing(eigenvalue_list, eigenbasis_projection_list)
    else:
        # Use ideal preprocessing (classical eigenvalue computation)
        print("     Using ideal preprocessing (classical eigenvalue computation)")
        e_preprocessing = ideal_preprocessing
    
    Enhanced_H_HHL = HHL(
        preprocessing=e_preprocessing,
        eigenvalue_inversion=EnhancedHybridInversion
    )
    
    # Create custom get_result function with projectors
    def get_result_with_projectors(circuit, prob):
        return get_braket_projection_result(circuit, prob, projectors)
    
    enhanced_result = Enhanced_H_HHL.estimate(
        problem=problem,
        num_clock_qubits=k_qubits,
        max_eigenvalue=10,
        quantum_conditional_logic=False,
        probability_threshold=probability_threshold,
        get_result_function=get_result_with_projectors
    )
    
    # Store results (convert complex numbers to real/imag pairs for JSON)
    ideal_solution_serializable = [
        {'real': float(x.real), 'imag': float(x.imag)} 
        for x in ideal_x.data
    ]
    
    enhanced_projection_results.append({
        'problem_index': i,
        'projection_tasks': enhanced_result.circuit_results,
        'depth': enhanced_result.depth,
        'num_projections': len(projectors),
        'ideal_solution': ideal_solution_serializable
    })
    
    print(f"     Circuit depth: {enhanced_result.depth}")
    print(f"     Submitted {len(projectors)} projection measurements")

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
    
    # Projection-specific data
    'projection_mode': PROJECTION_MODE,
    'projection_subset': PROJECTION_SUBSET if PROJECTION_MODE == "subset" else None,
    'num_solution_qubits': num_solution_qubits,
    'num_projections': len(projectors),
    'projector_basis_states': [p[0] for p in projectors],
    
    # Results
    'enhanced_projection_results': enhanced_projection_results,
}

# Save to file
file_name = f'braket_enhanced_projection_mat_{iteration}.json'
file_path = os.path.join(script_dir, file_name)

with open(file_path, "w") as json_file:
    json.dump(data, json_file, indent=2)

print(f"\n✓ Results saved to: {file_name}")

print("\n" + "="*60)
print("AWS Braket Enhanced HHL Projection Submission Complete!")
print("="*60)
print(f"\n{len(problem_list)} problems × {len(projectors)} projections")
print(f"Total tasks submitted: {len(problem_list) * len(projectors)}")
print("\nUse braket_projection_result_retrieval.py to retrieve and analyze results.")
