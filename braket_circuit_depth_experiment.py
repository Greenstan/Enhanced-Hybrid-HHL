"""
AWS Braket HHL Circuit Depth Experiment

Runs HHL with projection measurements on AWS Braket for different matrix sizes
and analyzes circuit depth, execution time, and solution accuracy.
"""

import os
import json
import numpy as np
import time
from datetime import datetime
from enhanced_hybrid_hhl import (
    QuantumLinearSystemProblem,
    run_braket_projection_submission,
    run_braket_projection_retrieval
)


def run_braket_circuit_depth_experiment(
    matrix_sizes=[2, 4, 8, 16],
    num_trials=20,
    k_qubits=7,
    shots=3000,
    preprocessing_mode="ideal",
    output_dir="./experiments/aws_braket",
    aws_region="eu-west-2",
    device_arn="SV1"
):
    """
    Run HHL circuit depth experiment on AWS Braket.
    
    Args:
        matrix_sizes: List of matrix dimensions to test
        num_trials: Number of trials per matrix size
        k_qubits: Number of clock qubits for QPE
        shots: Number of shots per projection measurement
        preprocessing_mode: "ideal", "lee", "yalovetzky", or "iterative"
        output_dir: Directory to save results
        aws_region: AWS region for Braket
        device_arn: Braket device ARN
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    
    print("="*70)
    print("AWS BRAKET HHL CIRCUIT DEPTH EXPERIMENT")
    print("="*70)
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Trials per size: {num_trials}")
    print(f"Total runs: {len(matrix_sizes) * num_trials}")
    print(f"Clock qubits: {k_qubits}")
    print(f"Shots per projection: {shots}")
    print(f"Preprocessing: {preprocessing_mode}")
    print(f"AWS Region: {aws_region}")
    print(f"Device: {device_arn}")
    print("="*70)
    print(f"{'N':<5} {'Trial':<7} {'Depth':<10} {'Projections':<12} {'Status'}")
    print("-"*70)
    
    # Storage for all results
    all_results = []
    submission_files = []
    
    for N in matrix_sizes:
        for trial in range(num_trials):
            try:
                # Generate random Hermitian matrix
                A_matrix = np.random.rand(N, N)
                A_matrix = A_matrix + A_matrix.T  # Make symmetric/Hermitian
                b_vector = np.random.rand(N)
                
                # Normalize b_vector
                b_vector = b_vector / np.linalg.norm(b_vector)
                
                # Calculate max eigenvalue for scaling
                eigenvalues = np.linalg.eigvalsh(A_matrix)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                
                # Create problem
                problem = QuantumLinearSystemProblem(A_matrix, b_vector)
                
                # Submit to Braket
                submission_start = time.time()
                # Ensure output_dir exists before calling submission
                os.makedirs(output_dir, exist_ok=True)

                result_path = run_braket_projection_submission(
                    problem_list=[problem],
                    output_dir=output_dir,
                    k_qubits=k_qubits,
                    shots=shots,
                    preprocessing_mode=preprocessing_mode,
                    aws_region=aws_region,
                    device_arn=device_arn,

                )
                submission_end = time.time()
                
                # Load submission results to get circuit depth
                with open(result_path, 'r') as f:
                    submission_data = json.load(f)
                
                circuit_depth = submission_data['enhanced_projection_results'][0]['depth']
                num_projections = submission_data['num_projections']
                
                # Store metadata
                trial_result = {
                    'matrix_size': N,
                    'trial': trial,
                    'circuit_depth': circuit_depth,
                    'num_projections': num_projections,
                    'submission_file': result_path,
                    'submission_time': submission_end - submission_start,
                    'max_eigenvalue': float(max_eigenvalue),
                    'timestamp': datetime.now().isoformat()
                }
                
                all_results.append(trial_result)
                submission_files.append(result_path)
                
                print(f"{N:<5} {trial+1:<7} {circuit_depth:<10} {num_projections:<12} {'✓'}")
                
            except Exception as e:
                print(f"{N:<5} {trial+1:<7} {'ERROR':<10} {'-':<12} {'✗'}")
                print(f"  Error: {str(e)[:60]}")
                
                trial_result = {
                    'matrix_size': N,
                    'trial': trial,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(trial_result)
    
    print("="*70)
    print("Submission phase complete!")
    print(f"Successfully submitted: {sum(1 for r in all_results if 'circuit_depth' in r)}/{len(all_results)}")
    print("="*70)
    
    # Save experiment metadata
    experiment_metadata = {
        'experiment_type': 'braket_hhl_circuit_depth',
        'parameters': {
            'matrix_sizes': matrix_sizes,
            'num_trials': num_trials,
            'k_qubits': k_qubits,
            'shots': shots,
            'preprocessing_mode': preprocessing_mode,
            'aws_region': aws_region,
            'device_arn': device_arn
        },
        'submission_results': all_results,
        'submission_files': submission_files,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata file
    metadata_filename = f"braket_experiment_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    print(f"\n✓ Experiment metadata saved to: {metadata_filename}")
    print(f"\nNext step: Retrieve results using retrieval script")
    print(f"  Submission files: {len(submission_files)}")
    
    return experiment_metadata


def retrieve_braket_experiment_results(
    metadata_file,
    aws_region="eu-west-2",
    device_arn="SV1",
    use_existing_retrieval=False
):
    """
    Retrieve all results from a Braket experiment using the metadata file.
    
    Args:
        metadata_file: Path to experiment metadata JSON file OR a single submission file
        aws_region: AWS region
        device_arn: Braket device ARN
        use_existing_retrieval: If True, use existing _retrieval files instead of fetching from AWS
        
    Returns:
        Dictionary with complete experiment results including fidelities and errors
    """
    print("="*70)
    if use_existing_retrieval:
        print("GATHERING RESULTS FROM EXISTING RETRIEVAL FILES")
    else:
        print("RETRIEVING BRAKET EXPERIMENT RESULTS")
    print("="*70)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check if this is an experiment metadata file or a single submission file
    if 'submission_files' in metadata:
        # This is an experiment metadata file
        submission_files = metadata['submission_files']
    else:
        # This is a single submission file - treat it as a single-file experiment
        submission_files = [metadata_file]
        metadata = {
            'experiment_type': 'single_retrieval',
            'submission_files': submission_files,
            'timestamp': datetime.now().isoformat()
        }
    
    print(f"Files to process: {len(submission_files)}")
    print("="*70)
    
    retrieval_results = []
    
    for idx, submission_file in enumerate(submission_files):
        print(f"\n[{idx+1}/{len(submission_files)}] Processing: {os.path.basename(submission_file)}")
        
        try:
            if use_existing_retrieval:
                # Look for existing retrieval file
                retrieval_path = submission_file.replace('.json', '_retrieval.json')
                
                if not os.path.exists(retrieval_path):
                    print(f"  ✗ Retrieval file not found: {os.path.basename(retrieval_path)}")
                    retrieval_results.append({
                        'submission_file': submission_file,
                        'error': 'Retrieval file not found'
                    })
                    continue
                
                print(f"  Using existing: {os.path.basename(retrieval_path)}")
            else:
                # Retrieve results from AWS
                retrieval_path = run_braket_projection_retrieval(
                    result_file_path=submission_file,
                    aws_region=aws_region,
                    device_arn=device_arn
                )
            
            # Load retrieval results
            with open(retrieval_path, 'r') as f:
                retrieval_data = json.load(f)
            
            result = retrieval_data['enhanced_projection_results'][0]
            
            # Extract execution times for each projection
            projection_execution_times = []
            for proj_task in result.get('projection_tasks', []):
                exec_time = proj_task.get('execution_time_seconds')
                projection_execution_times.append(exec_time)
            
            # Extract relevant metrics
            retrieval_result = {
                'submission_file': submission_file,
                'retrieval_file': retrieval_path,
                'fidelity': result.get('fidelity'),
                'circuit_depth': result['depth'],
                'num_projections': result['num_projections'],
                'average_success_rate': result['metadata']['average_success_rate'],
                'total_problem_time': result.get('total_problem_time_seconds'),
                'first_created_at': result.get('first_created_at'),
                'last_ended_at': result.get('last_ended_at'),
                'projection_execution_times': projection_execution_times,
                'reconstructed_solution': result.get('reconstructed_solution_probabilities'),
                'ideal_solution': result['ideal_solution']
            }
            
            retrieval_results.append(retrieval_result)
            
            print(f"  ✓ Fidelity: {retrieval_result['fidelity']:.4f}")
            if retrieval_result['total_problem_time']:
                print(f"  ✓ Total time: {retrieval_result['total_problem_time']:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}")
            retrieval_results.append({
                'submission_file': submission_file,
                'error': str(e)
            })
    
    print("\n" + "="*70)
    print("Retrieval phase complete!")
    print(f"Successfully retrieved: {sum(1 for r in retrieval_results if 'fidelity' in r)}/{len(retrieval_results)}")
    print("="*70)
    
    # Combine with original metadata
    metadata['retrieval_results'] = retrieval_results
    metadata['retrieval_timestamp'] = datetime.now().isoformat()
    
    # Save complete results
    output_dir = os.path.dirname(metadata_file)
    complete_filename = os.path.basename(metadata_file).replace('metadata', 'complete')
    complete_path = os.path.join(output_dir, complete_filename)
    
    with open(complete_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Complete experiment results saved to: {complete_filename}")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AWS Braket HHL Circuit Depth Experiment")
    parser.add_argument('--retrieve', type=str, metavar='METADATA_FILE',
                        help='Path to metadata file for retrieval phase')
    parser.add_argument('--use-existing', action='store_true',
                        help='Use existing _retrieval.json files instead of fetching from AWS')
    
    args = parser.parse_args()
    
    if args.retrieve:
        # Retrieval phase
        if args.use_existing:
            print("\nGathering results from existing retrieval files...\n")
        else:
            print("\nStarting retrieval phase...\n")
        
        complete_results = retrieve_braket_experiment_results(
            metadata_file=args.retrieve,
            use_existing_retrieval=args.use_existing
        )
        print("\n✓ Retrieval complete!")
        output_dir = os.path.dirname(args.retrieve)
        complete_filename = os.path.basename(args.retrieve).replace('metadata', 'complete')
        complete_path = os.path.join(output_dir, complete_filename)
        print("\nTo plot results, run:")
        print(f"  python braket_plot_results.py {complete_path}")
    else:
        # Submission phase (use defaults from function)
        print("\nStarting AWS Braket HHL Circuit Depth Experiment...\n")
        
        metadata = run_braket_circuit_depth_experiment(
            matrix_sizes=[2, 4, 8, 16],
            num_trials=20,
            k_qubits=7,
            shots=3000,
            preprocessing_mode="ideal",
            output_dir="./experiments/aws_braket"
        )
        
        print("\n✓ Experiment submission complete!")
        metadata_file = os.path.join("./experiments/aws_braket", 
                                     f"braket_experiment_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"\nTo retrieve results after tasks complete, run:")
        print(f"  python braket_circuit_depth_experiment.py --retrieve {metadata_file}")
