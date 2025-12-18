"""
Example: Using Braket Projection Utilities

Demonstrates how to use the run_braket_projection_submission and 
run_braket_projection_retrieval functions.
"""

import os
import json
import numpy as np
from enhanced_hybrid_hhl import (
    QuantumLinearSystemProblem,
    run_braket_projection_submission,
    run_braket_projection_retrieval
)

# Setup
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load problems from JSON
problem_file = os.path.join(script_dir, "custom_4x4_problem2.json")
with open(problem_file, 'r') as f:
    json_data = json.load(f)

# Convert to QuantumLinearSystemProblem objects
problem_list = []
for p in json_data['problem_list']:
    A_matrix = np.array(p['A_matrix'])
    b_vector = np.array(p['b_vector']).flatten()
    problem_list.append(QuantumLinearSystemProblem(A_matrix, b_vector))

print("="*60)
print("Example: Braket Projection Submission and Retrieval")
print("="*60 + "\n")

# ========== SUBMISSION ==========
print("STEP 1: Submitting projection measurements to AWS Braket\n")

result_file_path = run_braket_projection_submission(
    problem_list=problem_list,
    output_dir=script_dir,
    aws_region="eu-west-2",
    device_arn="SV1",
    k_qubits=10,
    shots=3000,
    preprocessing_mode="ideal",  # or "yalovetzky"
    projection_mode="all_components",  # or "subset"
    # projection_subset=[0, 1, 2],  # if using subset mode
    probability_threshold=0,
    max_eigenvalue=10,
    optimization_level=1
)

print(f"\n✓ Submission complete! Results saved to:\n  {result_file_path}\n")

# ========== RETRIEVAL ==========
print("\n" + "="*60)
print("STEP 2: Retrieving results from AWS Braket")
print("="*60 + "\n")

# You can retrieve immediately or later by providing the file path
retrieval_file_path = run_braket_projection_retrieval(
    result_file_path=result_file_path,
    aws_region="eu-west-2",
    device_arn="SV1",
    output_suffix="_retrieval"
)

print(f"\n✓ Retrieval complete! Results saved to:\n  {retrieval_file_path}\n")

print("="*60)
print("Done! Check the output JSON files for detailed results.")
print("="*60)
