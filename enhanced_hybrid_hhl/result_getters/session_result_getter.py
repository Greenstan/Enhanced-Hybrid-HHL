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

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
import numpy as np
from ..quantum_linear_system import QuantumLinearSystemProblem, HHL_Result
from .result_getter import result_getter
from .swaptest import SwapTest
from .swaptest_post_processing import st_post_processing
from qiskit_ibm_runtime import SamplerV2, Session
from qiskit.quantum_info import Statevector

class session_result_getter(result_getter):
    """
    The class `get_circuit_depth_result` takes an ibm backend as input and returns a function that
    transpiles a given quantum circuit and calculates its depth.
    
    :param backend: The `backend` parameter in the `get_circuit_depth_result` function is expected to be
    an object of type `Backend`.
    :type backend: Backend
    :return: A callable function `get_result` is being returned, which takes a `QuantumCircuit` and a
    `QuantumLinearSystemProblem` as input and returns an `HHL_Result` object.
    """
    def __init__(self,
                 session : Session,
                 statevector : Statevector):
        self.backend = session.service.get_backend(session.backend())
        self.sampler = SamplerV2(session=session)
        self.statevector = statevector

    def get_result(self,
                   hhl_circ : QuantumCircuit, 
                   problem : QuantumLinearSystemProblem,
                   ) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        hhl_circ.prepare_state(self.statevector, q_reg[:-1])
        hhl_circ.append(st, range(-st.num_qubits,0))
        hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, self.backend)
        
        job = self.sampler.run(circuit)

        result = HHL_Result()
        result.circuit_results = job.result().quasi_dists[0]
        result.results_processed = st_post_processing(result.circuit_results)
        result.job_id = job.job_id()
        result.circuit_depth = circuit.depth()
        return result
