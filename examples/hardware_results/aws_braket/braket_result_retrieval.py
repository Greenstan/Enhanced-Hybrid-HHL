import sys
import os
import json
import numpy as np
from typing import Dict, List

from qiskit_braket_provider import BraketProvider
import boto3

# Configuration
AWS_REGION = "eu-west-2"  # Must match the region used in submission
DEVICE_ARN = "SV1"

# Setup paths
script_dir = os.path.dirname(os.path.realpath(__file__))


def retrieve_braket_results(task_ids: List[str], backend) -> Dict:
    """
    Retrieve results from AWS Braket for a list of task IDs.
    
    Args:
        task_ids: List of AWS Braket task IDs
        backend: BraketBackend instance
        
    Returns:
        Dictionary mapping task_id to results
    """
    results = {}
    
    for task_id in task_ids:
        try:
            # Retrieve the job using the backend
            job = backend.retrieve_job(task_id)
            
            # Wait for completion if needed
            status = job.status()
            # Convert JobStatus enum to string for JSON serialization
            status_str = str(status.name) if hasattr(status, 'name') else str(status)
            print(f"Task {task_id}: {status_str}")
            
            # Check status (compare with enum values or strings)
            status_check = status_str if isinstance(status_str, str) else str(status)
            
            if status_check in ['COMPLETED', 'DONE', 'JobStatus.COMPLETED']:
                result = job.result()
                counts = result.get_counts()
                
                # Convert to probability distribution
                total = sum(counts.values())
                prob_dist = {key: value/total for key, value in counts.items()}
                
                results[task_id] = {
                    'status': status_str,
                    'counts': counts,
                    'probabilities': prob_dist,
                    'shots': total
                }
            elif status_check in ['FAILED', 'CANCELLED', 'JobStatus.FAILED', 'JobStatus.CANCELLED']:
                results[task_id] = {
                    'status': status_str,
                    'error': 'Task failed or was cancelled'
                }
            else:
                results[task_id] = {
                    'status': status_str,
                    'message': 'Task still running or queued'
                }
                
        except Exception as e:
            print(f"Error retrieving task {task_id}: {e}")
            results[task_id] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    return results


def load_submission_data(filename: str) -> Dict:
    """Load submission data from JSON file."""
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'r') as file:
        return json.load(file)


def save_results(data: Dict, filename: str):
    """Save results to JSON file."""
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"✓ Results saved to: {filename}")


def main():
    """Main function to retrieve and save AWS Braket results."""
    
    # Setup AWS Braket provider and backend
    boto_session = boto3.Session(region_name=AWS_REGION)
    provider = BraketProvider()
    backend = provider.get_backend(DEVICE_ARN)
    
    print("="*60)
    print("AWS Braket Result Retrieval")
    print("="*60 + "\n")
    print(f"Using backend: Simulator")
    print(f"Region: {AWS_REGION}\n")
    
    # Find all submission files
    submission_files = [f for f in os.listdir(script_dir) 
                       if f.startswith('braket_enhanced') 
                       and f.endswith('.json')]
    
    if not submission_files:
        print("No submission files found!")
        print(f"Looking in: {script_dir}")
        return
    
    print(f"Found {len(submission_files)} submission file(s)\n")
    
    for submission_file in submission_files:
        print(f"\nProcessing: {submission_file}")
        
        # Load submission data
        data = load_submission_data(submission_file)
        
        # Collect all task IDs
        all_task_ids = []
        task_id_mapping = {}
        
        if 'canonical_ids' in data:
            for idx, task_id in enumerate(data['canonical_ids']):
                all_task_ids.append(task_id)
                task_id_mapping[task_id] = ('canonical', idx)
        
        if 'hybrid_ids' in data:
            for idx, task_id in enumerate(data['hybrid_ids']):
                all_task_ids.append(task_id)
                task_id_mapping[task_id] = ('hybrid', idx)
        
        if 'enhanced_ids' in data:
            for idx, task_id in enumerate(data['enhanced_ids']):
                all_task_ids.append(task_id)
                task_id_mapping[task_id] = ('enhanced', idx)
        
        print(f"  Total tasks: {len(all_task_ids)}")
        
        # Retrieve results
        print("\nRetrieving results...")
        results = retrieve_braket_results(all_task_ids, backend)
        
        # Update data with results
        canonical_results = []
        hybrid_results = []
        enhanced_results = []
        
        completed = 0
        failed = 0
        pending = 0
        
        for task_id, result in results.items():
            method, idx = task_id_mapping[task_id]
            
            status = result.get('status', 'UNKNOWN')
            if status in ['COMPLETED', 'DONE']:
                completed += 1
            elif status in ['FAILED', 'CANCELLED', 'ERROR']:
                failed += 1
            else:
                pending += 1
            
            if method == 'canonical':
                canonical_results.append(result)
            elif method == 'hybrid':
                hybrid_results.append(result)
            elif method == 'enhanced':
                enhanced_results.append(result)
        
        # Update data dictionary
        data['canonical_results'] = canonical_results
        data['hybrid_results'] = hybrid_results
        data['enhanced_results'] = enhanced_results
        data['retrieval_summary'] = {
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'total': len(all_task_ids)
        }
        
        # Save updated data
        output_filename = submission_file.replace('braket_enhanced', 'braket_results_custom')
        save_results(data, output_filename)
        
        # Print summary
        print(f"\nResults Summary:")
        print(f"  ✓ Completed: {completed}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⏳ Pending: {pending}")
        print(f"  Total: {len(all_task_ids)}")
    
    print("\n" + "="*60)
    print("Result Retrieval Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
