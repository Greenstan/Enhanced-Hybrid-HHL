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

from qiskit import QuantumCircuit, transpile
from ..quantum_linear_system import QuantumLinearSystemProblem, HHL_Result, QuantumLinearSystemSolver
from .result_getter import result_getter
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator
import numpy as np

class fidelity_result_getter(result_getter):
    """
    The class `get_circuit_depth_result` takes an ibm backend as input and returns a function that
    transpiles a given quantum circuit and calculates its depth.
    
    :param backend: The `backend` parameter in the `get_circuit_depth_result` function is expected to be
    an object of type `Backend`.
    :type backend: Backend
    :return: A callable function `get_result` is being returned, which takes a `QuantumCircuit` and a
    `QuantumLinearSystemProblem` as input and returns an `HHL_Result` object.
    """
    def get_result(self,
                   hhl_circ : QuantumCircuit, 
                   problem : QuantumLinearSystemProblem,
                   ) -> HHL_Result:
        r'''Function to simulate the hhl_circuit, and return the inner product between the simulated estimate
        of |x> and the classically calculated solution.
        Args:
            '''

        simulator = AerSimulator()
        # create projection operator of classically calculated solution
        ideal_x_operator = QuantumLinearSystemSolver(problem).ideal_x_statevector.to_operator()

        # qubits to remove in the partial trace of the simulated statevector. 
        trace_qubits = list(range(hhl_circ.num_qubits-ideal_x_operator.num_qubits))

        # save statevector if the inversion was successful
        with hhl_circ.if_test((0,1)) as passed:
            hhl_circ.save_state()

        # transpile circuit
        circ = transpile(hhl_circ, simulator)
        simulated_result = simulator.run(circ).result()

        # show the success probability of inversion
        circuit_results = simulated_result.get_counts()

        # retrieve statevector
        simulated_statevector = simulated_result.get_statevector()
        partial_simulated_density_matrix = partial_trace(simulated_statevector, trace_qubits)

        # square root of the expectation value is the fidelity
        result_processed = np.sqrt(partial_simulated_density_matrix.expectation_value(ideal_x_operator))

        result = HHL_Result()
        result.circuit_results = circuit_results
        result.results_processed = result_processed
        result.circuit_depth = circ.depth()