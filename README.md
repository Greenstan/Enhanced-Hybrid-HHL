# Enhanced Hybrid HHL (Jaffer)

Project used to test Publicly Verifiable Quantum Computation Scheme with a Time Locked Puzzle and a simultaneous Hybrid Harrow-Hassidim-LLoyd (HHL) algo. 

## Relevant newly added Files/directories
/puzzle-generations - all the code and funcs used to create/solve TLP 
HHL_example.ipynb - Jupyter notebook sandbox to run Enhanced Hybrid HHL. Important functions:
        *hermitianize_matrix* - take non-hermitian matrix and hermitianize + pad it
        *FixedYalovetzkyPreprocessing* - a fixed version of the Hybrid Preprocessing QPE algorithm 
        *create_projection_operator* - Create a single non-zero element projection operator for HHL measurement
        *solve_linear_system_with_hhl* - Full Enhanced Hybrid HHL (dynamic size) algo with multiple configurations
        *visualize_hhl_results* - create mpl graphs of results
experiment.ipynb - CGM algorithm and puzzle generation and solving tests.
/examples/hardware_results/aws_braket/braket_enhanced_projection_submission.py - AWS Braket Statevector submission for HHL algo ( creates a JSON file with quantum task ID )
/examples/hardware_results/aws_braket/braket_enhanced_projection_retrieval.py - Uses submission Quantum task Ids to retrieve solutions and output them (then creates a JSON file with results)



Todo:
[ ] Create function for enhanced_projection_submission/retrieval to be called in Jupyter notebook
[ ] Remove old unused experimentation code in HHL_example.ipynb



## Enhanced-Hybrid-HHL (Legacy Docs)
This project contains code and example problems used to the Enhanced Hybrid HHL Algorithm [1]. For benchmarking purposes, the project can implement the standard HHL algorithm [2], the Hybrid HHL algorithm [3], as well as the variation of the Hybrid HHL algorithm proposed by [4]. The variant of the HHL class is determined by the choice of inversion circuit and 
eigenvalue preprocessing parameter. 

## Example Implementation 
### Step one: Define a Quantum Linear System Problem
In this example we estimate $\ket{x}$ in the equation $\mathcal{A} \ket{x} = \ket{b}$
where $\mathcal{A} = $ A_matrix and $\ket{b} =$ b_vector defined below. We test the accuracy
of the algorithm by observing the estimated state $\ket{x}$ with the projection operator 
onto the ideal solution.

```python
from enhanced_hybrid_hhl import (HHL, 
                                 Lee_preprocessing,  
                                 HybridInversion, 
                                 QuantumLinearSystemProblem, 
                                 QuantumLinearSystemSolver,
                                 EnhancedHybridInversion)
import numpy as np
from qiskit_aer import AerSimulator

# define the backend to run the circuits on
simulator = AerSimulator()

# Define quantum linear system problem to be solved with HHL
a_matrix = np.array([[ 0.5 , -0.25],
        [-0.25,  0.5 ]])
b_vector = np.array([[1.], [0.]])
problem = QuantumLinearSystemProblem(A_matrix=a_matrix,
                                     b_vector=b_vector)
```
### Step two: Choose the algorithm parameters
```python
k = 3 # clock qubits for hhl.
l = k+2 # clock qubits for enhanced preprocessing.
min_prob = 2**-k # hybrid preprocessing relevance threshold.
relevance_threshold = 2**-l # enhanced hybrid preprocessing relevance threshold.
maximum_eigenvalue = 1 # Over estimate of largest eigenvalue in the system.

get_result_type = 'get_swap_test_result'
ideal_x_statevector = QuantumLinearSystemSolver(problem=problem).ideal_x_statevector
```

### Step three: Define Preprocessing and Inversion circuit classes
```python
# In this example, we use the standard QPEA used by Lee et al.
enhanced_preprocessing = Lee_preprocessing(num_eval_qubits=l,
                                  max_eigenvalue= maximum_eigenvalue, 
                                  backend=simulator).estimate

enhanced_eigenvalue_inversion = EnhancedHybridInversion
```
### Step four: Create the HHL Class
```python
enhanced_hybrid_hhl = HHL(get_result_function= get_result_type,
          preprocessing= enhanced_preprocessing,
          eigenvalue_inversion= enhanced_eigenvalue_inversion,
          backend=simulator,
          statevector=ideal_x_statevector)
```
### Step five: Run the algorithm
```python
enhanced_hybrid_hhl_result = enhanced_hybrid_hhl.estimate(problem=problem,
                                                          num_clock_qubits=k,
                                                          max_eigenvalue=1)

print(enhanced_hybrid_hhl_result)
```
### References
<a id="1">[1]</a> 
Morgan, J., Ghysels, E., & Mohammadbagherpoor, H. (2024). An Enhanced Hybrid HHL Algorithm. arXiv preprint arXiv:2404.10103.

<a id="2">[2]</a> 
Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear systems of equations. Physical review letters, 103(15), 150502.

<a id="3">[3]</a> 
Lee, Y., Joo, J., & Lee, S. (2019). Hybrid quantum linear equation algorithm and its experimental test on IBM Quantum Experience. Scientific reports, 9(1), 4778.

<a id="4">[4]</a> 
Yalovetzky, R., Minssen, P., Herman, D., & Pistoia, M. (2021). Hybrid HHL with Dynamic Quantum Circuits on Real Hardware. arXiv preprint arXiv:2110.15958.

## License

This project uses [Apache 2.0 License]([url](https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)









