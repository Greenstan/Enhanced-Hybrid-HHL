# AWS Braket HHL Circuit Depth Experiment

This directory contains scripts for running comprehensive HHL circuit depth experiments on AWS Braket and visualizing the results.

## Files

- **`braket_circuit_depth_experiment.py`**: Main experiment script for submission and retrieval
- **`braket_plot_results.py`**: Plotting and analysis functions
- **`EXPERIMENT_README.md`**: This file

## Quick Start

### 1. Run the Experiment

```python
from braket_circuit_depth_experiment import run_braket_circuit_depth_experiment

# Submit experiments to AWS Braket
metadata = run_braket_circuit_depth_experiment(
    matrix_sizes=[2, 4, 8, 16],  # Matrix dimensions to test
    num_trials=20,                # Trials per matrix size
    k_qubits=7,                   # Clock qubits for QPE
    shots=3000,                   # Shots per projection
    preprocessing_mode="ideal",   # "ideal", "lee", "yalovetzky", or "iterative"
    output_dir="./examples/hardware_results/aws_braket"
)
```

### 2. Retrieve Results

```python
from braket_circuit_depth_experiment import retrieve_braket_experiment_results

# Retrieve all quantum task results
complete_results = retrieve_braket_experiment_results(
    metadata_file="./examples/hardware_results/aws_braket/braket_experiment_metadata_YYYYMMDD_HHMMSS.json"
)
```

### 3. Plot and Analyze

```python
from braket_plot_results import plot_braket_experiment_results

# Generate comprehensive visualizations
stats = plot_braket_experiment_results(
    complete_results_file="./examples/hardware_results/aws_braket/braket_experiment_complete_YYYYMMDD_HHMMSS.json",
    show_plots=True,
    save_plots=True
)
```

Or from command line:
```bash
python braket_plot_results.py ./examples/hardware_results/aws_braket/braket_experiment_complete_*.json
```

## Experiment Workflow

### Phase 1: Submission

1. Generates random Hermitian matrices for each size and trial
2. Creates `QuantumLinearSystemProblem` instances
3. Submits projection measurements to AWS Braket using `run_braket_projection_submission()`
4. Records circuit depth, number of projections, and submission metadata
5. Saves experiment metadata JSON file

**Output:**
- Individual submission JSON files for each problem
- Master metadata file: `braket_experiment_metadata_<timestamp>.json`

### Phase 2: Retrieval

1. Loads experiment metadata
2. Retrieves results for all quantum tasks using `run_braket_projection_retrieval()`
3. Extracts fidelity, success rates, and timing information
4. Combines submission and retrieval data

**Output:**
- Individual retrieval JSON files
- Complete results file: `braket_experiment_complete_<timestamp>.json`

### Phase 3: Analysis

1. Aggregates results by matrix size
2. Calculates statistics (mean, std dev) for each metric
3. Generates 8-panel visualization:
   - Circuit depth (linear and log scale)
   - Execution time (linear and log scale)
   - Solution fidelity
   - Measurement success rate
   - Depth per qubit
   - Summary statistics
4. Performs linear regression for growth rate analysis

**Output:**
- PNG plot file: `braket_experiment_complete_<timestamp>_plots.png`
- Console summary statistics

## Experiment Parameters

### Matrix Sizes
- Default: `[2, 4, 8, 16]`
- Corresponds to Hilbert space dimensions: 2, 4, 8, 16
- Solution qubits required: 1, 2, 3, 4

### Trials per Size
- Default: `20`
- More trials → better statistics but longer runtime
- Recommended minimum: 10

### Clock Qubits (k_qubits)
- Default: `7`
- Higher values → better eigenvalue resolution
- More qubits → deeper circuits
- Typical range: 6-10

### Shots per Projection
- Default: `3000`
- Higher shots → better measurement statistics
- Trade-off: cost vs accuracy
- Typical range: 1000-10000

### Preprocessing Mode
- **`"ideal"`** (default): Classical eigenvalue computation (fastest, most accurate)
- **`"lee"`**: Standard QPE (Braket-compatible)
- **`"yalovetzky"`**: QCL-QPE (NOT Braket-compatible - requires reset gates)
- **`"iterative"`**: Iterative QPE (NOT Braket-compatible - requires reset gates)

## Results Interpretation

### Circuit Depth
- **Linear growth**: Indicates polynomial scaling
- **Exponential growth**: May indicate challenges for NISQ devices
- **Depth per qubit**: Normalized metric for comparing different sizes

### Fidelity
- Measures overlap between ideal and measured solution
- Range: [0, 1], where 1 is perfect
- Expected degradation with matrix size due to:
  - Increased circuit depth
  - More complex eigenvalue structure
  - Finite shot statistics

### Success Rate
- Fraction of measurements with ancilla qubit = 1
- Indicates eigenvalue inversion quality
- Lower success rate → more shots needed

### Execution Time
- Wall-clock time from first task creation to last task completion
- Accounts for parallel execution on AWS Braket
- Includes queueing, execution, and post-processing

## Example Output

```
======================================================================
AWS BRAKET HHL CIRCUIT DEPTH EXPERIMENT
======================================================================
Matrix sizes: [2, 4, 8, 16]
Trials per size: 20
Total runs: 80
Clock qubits: 7
Shots per projection: 3000
Preprocessing: ideal
AWS Region: eu-west-2
Device: SV1
======================================================================
N     Trial   Depth      Projections  Status
----------------------------------------------------------------------
2     1       94         2            ✓
2     2       94         2            ✓
...
16    20      6387       16           ✓
======================================================================
Submission phase complete!
Successfully submitted: 80/80
======================================================================
```

## Cost Estimation

AWS Braket SV1 pricing (as of 2025):
- ~$0.075 per task

For the default experiment:
- Matrix sizes: 4 (2, 4, 8, 16)
- Trials: 20
- Projections per problem: varies (2, 4, 8, 16)
- Total tasks: ~600
- **Estimated cost: ~$45**

## Tips

1. **Start small**: Test with 1-2 trials first to verify everything works
2. **Monitor costs**: Check AWS Braket console for real-time task counts
3. **Save metadata**: Keep all metadata files for reproducibility
4. **Incremental retrieval**: You can retrieve results in batches
5. **Error handling**: Failed tasks are logged; re-run only failed trials if needed

## Troubleshooting

**Problem**: Tasks fail with "Device requires all qubits measured"
- **Solution**: This is handled automatically; check if using correct preprocessing mode

**Problem**: Yalovetzky preprocessing doesn't work
- **Solution**: Use "ideal" or "lee" preprocessing for AWS Braket

**Problem**: Results show very low fidelity
- **Solution**: Increase shots, check if eigenvalues are well-conditioned

**Problem**: Retrieval times out
- **Solution**: Wait for all tasks to complete; check AWS Braket console

## Advanced Usage

### Custom Problem Generation

```python
# Instead of random matrices, use specific problems
from enhanced_hybrid_hhl import QuantumLinearSystemProblem
import numpy as np

# Create well-conditioned problem
A = np.diag([2, 3, 5, 7])  # Diagonal matrix
b = np.array([1, 1, 1, 1])
problem = QuantumLinearSystemProblem(A, b)

# Submit single problem
result_path = run_braket_projection_submission(
    problem_list=[problem],
    output_dir="./results",
    k_qubits=7,
    shots=3000
)
```

### Batch Processing

```python
# Process multiple experiments
metadata_files = glob.glob("./results/braket_experiment_metadata_*.json")

for metadata_file in metadata_files:
    complete_results = retrieve_braket_experiment_results(metadata_file)
    plot_braket_experiment_results(complete_results)
```

## Citation

If you use this experiment framework, please cite:

```bibtex
@software{enhanced_hybrid_hhl,
  title={Enhanced Hybrid HHL with AWS Braket},
  author={Your Name},
  year={2025},
  url={https://github.com/Greenstan/Enhanced-Hybrid-HHL}
}
```
