#!/usr/bin/env python3
"""
Test script to verify AWS Braket setup before running full experiments.
"""

import sys
import boto3
from qiskit import QuantumCircuit, transpile
from qiskit_braket_provider import BraketProvider

def test_aws_credentials():
    """Test if AWS credentials are properly configured."""
    print("Testing AWS Credentials...")
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            print("  ✓ AWS credentials found")
            print(f"    Access Key: {credentials.access_key[:8]}...")
            print(f"    Region: {session.region_name or 'Not set (will use default)'}")
            return True
        else:
            print("  ✗ No AWS credentials found")
            print("\n  Configure credentials using:")
            print("    1. aws configure")
            print("    2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars")
            print("    3. Create ~/.aws/credentials file")
            return False
    except Exception as e:
        print(f"  ✗ Error checking credentials: {e}")
        return False


def test_braket_provider():
    """Test if BraketProvider can be initialized."""
    print("\nTesting BraketProvider...")
    try:
        provider = BraketProvider()
        print("  ✓ BraketProvider initialized successfully")
        return provider
    except Exception as e:
        print(f"  ✗ Error initializing BraketProvider: {e}")
        return None


def test_backend_access(provider, device_name="SV1"):
    """Test if we can access a Braket backend."""
    print(f"\nTesting Backend Access ({device_name})...")
    try:
        backend = provider.get_backend(device_name)
        print(f"  ✓ Successfully connected to {backend.name()}")
        print(f"    Backend type: {backend.backend_type}")
        
        # Try to get backend properties
        try:
            print(f"    Max shots: {backend.max_shots}")
        except:
            print(f"    Max shots: N/A")
        
        return backend
    except Exception as e:
        print(f"  ✗ Error accessing backend: {e}")
        return None


def test_simple_circuit(backend):
    """Test running a simple circuit on the backend."""
    print(f"\nTesting Simple Circuit Execution...")
    try:
        # Create a simple Bell state circuit
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        print("  Created Bell state circuit")
        
        # Transpile
        transpiled = transpile(circuit, backend)
        print(f"  Transpiled circuit (depth: {transpiled.depth()})")
        
        # Run with minimal shots for testing
        print("  Submitting job to AWS Braket...")
        job = backend.run(transpiled, shots=100)
        job_id = job.job_id()
        print(f"  ✓ Job submitted successfully")
        print(f"    Job ID: {job_id}")
        
        # Try to get status
        print(f"  Waiting for results...")
        result = job.result()
        counts = result.get_counts()
        
        print(f"  ✓ Results received:")
        for state, count in sorted(counts.items()):
            print(f"    {state}: {count}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error running circuit: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_measurement_requirement(backend):
    """Test that all qubits must be measured (Braket requirement)."""
    print(f"\nTesting Measurement Requirement...")
    try:
        # Create circuit with unmeasured qubit (should fail on AWS Braket)
        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)  # Only measure 1 qubit
        
        transpiled = transpile(circuit, backend)
        
        print("  Submitting circuit with partial measurements...")
        job = backend.run(transpiled, shots=10)
        
        try:
            result = job.result()
            print("  ⚠ Warning: Backend accepted partial measurements")
            print("    (This is OK for local simulator, but would fail on AWS SV1)")
        except Exception as e:
            if "all qubits" in str(e).lower() or "measured" in str(e).lower():
                print("  ✓ Backend correctly requires all qubits to be measured")
                print("    (This is expected for AWS Braket)")
            else:
                raise
        
        return True
        
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("AWS Braket Setup Test")
    print("="*60)
    
    # Test 1: Credentials
    if not test_aws_credentials():
        print("\n" + "="*60)
        print("❌ FAILED: AWS credentials not configured")
        print("="*60)
        sys.exit(1)
    
    # Test 2: Provider
    provider = test_braket_provider()
    if not provider:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot initialize BraketProvider")
        print("="*60)
        sys.exit(1)
    
    # Test 3: Backend access
    backend = test_backend_access(provider, "SV1")
    if not backend:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot access backend")
        print("="*60)
        print("\nNote: If using AWS SV1, ensure:")
        print("  1. Your AWS account has Braket enabled")
        print("  2. You have proper IAM permissions")
        print("  3. Braket is available in your region")
        sys.exit(1)
    
    # Test 4: Circuit execution
    if not test_simple_circuit(backend):
        print("\n" + "="*60)
        print("❌ FAILED: Cannot execute circuits")
        print("="*60)
        sys.exit(1)
    
    # Test 5: Measurement requirement (informational)
    test_measurement_requirement(backend)
    
    # All tests passed
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nYour AWS Braket setup is working correctly!")
    print("You can now run braket_submission_script.py")
    print("="*60)


if __name__ == "__main__":
    main()
