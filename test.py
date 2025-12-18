import os
import numpy as np
from enhanced_hybrid_hhl import (
    QuantumLinearSystemProblem,
    run_braket_projection_submission,
    run_braket_projection_retrieval
)

# # Create custom 4x4 problem
A_matrix = np.array([
    [2, 0, 0, 4],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [4, 0, 0, 2]
])

b_vector = np.array([1.0, 2.0, 0.0, 1.5])

# problem1 = QuantumLinearSystemProblem(A_matrix, b_vector)

# # Create another problem
# A_matrix2 = np.array([
#     [3, 1, 0, 0],
#     [1, 3, 0, 0],
#     [0, 0, 3, 1],
#     [0, 0, 1, 3]
# ])

# b_vector2 = np.array([1.0, 0.0, 1.0, 0.0])

# problem2 = QuantumLinearSystemProblem(A_matrix2, b_vector2)

# Create custom 16x16 sparse Hermitian problem
# Use a sparse tridiagonal-like structure with some additional off-diagonal elements
# A_matrix = np.zeros((16, 16))

# # Main diagonal
# for i in range(16):
#     A_matrix[i, i] = 4.0

# # First off-diagonal (upper and lower) - Hermitian symmetric
# for i in range(15):
#     A_matrix[i, i+1] = 1.0
#     A_matrix[i+1, i] = 1.0

# # Add some sparse coupling elements (Hermitian symmetric)
# A_matrix[0, 4] = 0.5
# A_matrix[4, 0] = 0.5
# A_matrix[3, 7] = 0.5
# A_matrix[7, 3] = 0.5
# A_matrix[8, 12] = 0.5
# A_matrix[12, 8] = 0.5
# A_matrix[11, 15] = 0.5
# A_matrix[15, 11] = 0.5

# # Create b vector with sparse structure
# b_vector = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 
#                      1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0])

problem1 = QuantumLinearSystemProblem(A_matrix, b_vector)

# Setup output directory
script_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(script_dir, "examples/hardware_results/aws_braket")

# Submit
print("Submitting projection measurements to AWS Braket...")
result_path = run_braket_projection_submission(
    problem_list=[problem1],
    output_dir=output_dir,
    k_qubits=7,
    shots=3000,
    preprocessing_mode="lee"
)

# result_path = "/Users/jaffermahdi/Desktop/Development/Enhanced-Hybrid-HHL/examples/hardware_results/aws_braket/braket_projection_N4_hhl_20251218_131807.json"


print(f"\nSubmission complete! Results file: {result_path}")

# Retrieve (now or later)
print("\nRetrieving results from AWS Braket...")
retrieval_path = run_braket_projection_retrieval(
    result_file_path=result_path
)

print(f"\nRetrieval complete! Results file: {retrieval_path}")