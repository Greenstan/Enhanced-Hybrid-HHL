# AWS Braket HHL Submission Scripts

This directory contains scripts for running HHL experiments on AWS Braket quantum devices and simulators.

## Files

- **`braket_submission_script.py`** - Submit HHL circuits to AWS Braket
- **`braket_result_retrieval.py`** - Retrieve results from submitted tasks
- **`README.md`** - This file

## Prerequisites

### 1. AWS Account Setup

You need an AWS account with Braket access:
1. Create an AWS account at https://aws.amazon.com/
2. Enable AWS Braket service
3. Create an IAM user with Braket permissions or use root credentials

### 2. AWS Credentials Configuration

Configure AWS credentials using one of these methods:

**Option A: AWS CLI Configuration (Recommended)**
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (e.g., eu-west-2)
# Enter default output format (json)
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=eu-west-2
```

**Option C: Credentials File**
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

And `~/.aws/config`:
```ini
[default]
region = eu-west-2
```

### 3. Python Dependencies

Install required packages:
```bash
pip install qiskit qiskit-braket-provider boto3 amazon-braket-sdk
pip install -e /path/to/enhanced_hybrid_hhl  # Your HHL package
```

### 4. S3 Bucket (Optional but Recommended)

Create an S3 bucket for storing results:
```bash
aws s3 mb s3://your-hhl-results-bucket --region eu-west-2
```

Update `braket_submission_script.py`:
```python
S3_BUCKET = "your-hhl-results-bucket"
S3_PREFIX = "hhl-results"
```

## Usage

### Step 1: Prepare Preprocessing Data

Ensure you have a preprocessing file (e.g., `simulator_small_matrix_preprocessing.json`) in the same directory. This should contain:
- `lam_list`: List of problem parameters
- `fixed`: Preprocessing results for hybrid method
- `enhanced_fixed`: Preprocessing results for enhanced hybrid method

### Step 2: Configure Backend

Edit `braket_submission_script.py` to set your desired backend:

```python
# Configuration
AWS_REGION = "eu-west-2"  # Your AWS region
DEVICE_ARN = "SV1"        # Options below
S3_BUCKET = "your-bucket" # Your S3 bucket (optional)
```

**Available Backends:**
- `"SV1"` - State vector simulator (up to 34 qubits)
- `"TN1"` - Tensor network simulator (up to 50 qubits)
- `"dm1"` - Density matrix simulator (noise simulation)
- Device ARN for quantum hardware (e.g., `"arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"`)

### Step 3: Submit Jobs

Run the submission script:
```bash
python braket_submission_script.py
```

This will:
1. Load problem configurations
2. Run Canonical, Hybrid, and Enhanced Hybrid HHL
3. Submit circuits to AWS Braket
4. Save task IDs to JSON files (e.g., `simulator_to_braket_N2_matrix_hhl3.json`)

**Output:**
```
============================================================
Starting AWS Braket HHL Submission
============================================================

--- Iteration 3 ---

Processing problem 1/10...
  → Running Canonical HHL...
     Task ID: arn:aws:braket:eu-west-2:123456789:quantum-task/abc123...
  → Running Hybrid HHL...
     Task ID: arn:aws:braket:eu-west-2:123456789:quantum-task/def456...
  → Running Enhanced Hybrid HHL...
     Task ID: arn:aws:braket:eu-west-2:123456789:quantum-task/ghi789...
...

✓ Results saved to: simulator_to_braket_N2_matrix_hhl3.json
```

### Step 4: Retrieve Results

After submitting jobs, retrieve the results:
```bash
python braket_result_retrieval.py
```

This will:
1. Find all submission JSON files
2. Retrieve results for each task ID
3. Save complete results to new files (e.g., `braket_results_N2_matrix_hhl3.json`)

**Output:**
```
============================================================
AWS Braket Result Retrieval
============================================================

Found 3 submission file(s)

Processing: simulator_to_braket_N2_matrix_hhl3.json
  Total tasks: 30
Retrieving results...
Task arn:aws:braket:...:abc123: COMPLETED
Task arn:aws:braket:...:def456: COMPLETED
...

Results Summary:
  ✓ Completed: 28
  ✗ Failed: 0
  ⏳ Pending: 2
  Total: 30

✓ Results saved to: braket_results_N2_matrix_hhl3.json
```

## Result Format

The output JSON files contain:

```json
{
  "problem_list": [...],
  "shots": 1000,
  "backend": "SV1",
  "device_arn": "SV1",
  "aws_region": "eu-west-2",
  
  "canonical_ids": ["task_id_1", "task_id_2", ...],
  "canonical_depths": [245, 267, ...],
  "canonical_results": [
    {
      "status": "COMPLETED",
      "counts": {"00": 523, "01": 477},
      "probabilities": {"00": 0.523, "01": 0.477},
      "shots": 1000
    },
    ...
  ],
  
  "hybrid_ids": [...],
  "hybrid_results": [...],
  
  "enhanced_ids": [...],
  "enhanced_results": [...]
}
```

## Cost Considerations

### Simulators
- **SV1**: Free tier available, then ~$0.075 per minute
- **TN1**: ~$0.275 per minute
- **DM1**: ~$0.075 per minute

### Quantum Hardware
- **IonQ**: ~$0.30 per task + $0.01 per shot
- **Rigetti**: ~$0.30 per task + $0.00035 per shot
- Varies by device and availability

**Estimate costs before running large experiments!**

Use the [AWS Braket Pricing Calculator](https://calculator.aws/#/addService/Braket).

## Troubleshooting

### Error: "ValidationException: Device requires all qubits to be measured"

The script includes `ensure_all_qubits_measured()` to fix this automatically. If you still see this error:
- Check that your HHL circuits use contiguous qubit indices starting from 0
- Ensure all qubits are being measured

### Error: "NoCredentialsError"

Your AWS credentials are not configured. See Prerequisites section above.

### Error: "AccessDeniedException"

Your AWS account doesn't have permission to use Braket. Contact your AWS administrator.

### Tasks stuck in "QUEUED" status

Hardware devices may have long queue times. Simulators are usually fast (seconds to minutes).

### Results not available

Tasks may take time to complete, especially on hardware. Run `braket_result_retrieval.py` again later.

## Monitoring

Monitor your tasks in the AWS Console:
1. Go to https://console.aws.amazon.com/braket/
2. Navigate to "Quantum Tasks"
3. View task status, costs, and results

## Advanced Configuration

### Custom Get Result Function

Modify `get_braket_result()` in `braket_submission_script.py` to customize:
- Measurement strategy
- Shot count
- Post-processing

### Running on Quantum Hardware

Change `DEVICE_ARN` to a hardware device:
```python
# IonQ Harmony
DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"

# Rigetti Aspen-M-3
DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"

# OQC Lucy
DEVICE_ARN = "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy"
```

**Note:** Hardware devices have specific:
- Availability windows
- Qubit connectivity constraints
- Gate sets
- Error rates

Consult AWS Braket documentation for device specifications.

## References

- [AWS Braket Documentation](https://docs.aws.amazon.com/braket/)
- [Qiskit Braket Provider](https://github.com/qiskit-community/qiskit-braket-provider)
- [Enhanced Hybrid HHL](https://github.com/Greenstan/Enhanced-Hybrid-HHL)

## Support

For issues related to:
- **AWS Braket**: AWS Support or Braket forums
- **HHL Implementation**: Open an issue on the Enhanced-Hybrid-HHL repository
- **Qiskit-Braket Provider**: https://github.com/qiskit-community/qiskit-braket-provider/issues
