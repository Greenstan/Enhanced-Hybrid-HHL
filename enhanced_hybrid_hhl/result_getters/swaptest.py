'''
 Copyright 2023 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from qiskit import QuantumCircuit

def SwapTest(num_state_qubits : int)-> QuantumCircuit:
    """
    The function `SwapTest` creates a quantum circuit that performs a swap test between two sets of
    qubits.
    
    :param num_state_qubits: The `num_state_qubits` parameter in the `SwapTest` function represents the
    number of qubits used to encode the state you want to test. This function creates a quantum circuit
    that performs a swap test between the reference state encoded in the last qubit and the input state
    encoded in the first
    :return: The function `SwapTest(num_state_qubits)` returns a quantum circuit `st_circ` that performs
    a swap test between the state of the last qubit and the state of the first `num_state_qubits`
    qubits.
    """
    num_qubits = 2*num_state_qubits+1
    st_circ = QuantumCircuit(num_qubits)
    st_circ.h(-1)
    for i in range(num_state_qubits):
        st_circ.cswap(-1,i,num_state_qubits+i)
    st_circ.h(-1)
    return st_circ
