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

import __future__

from ..quantum_linear_system import QuantumLinearSystemSolver, QuantumLinearSystemProblem
from ..result_getters import preprocessing_result_getter, preprocessing_backend_result_getter, preprocessing_session_result_getter
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library import PhaseEstimation, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.providers import Backend
#from qiskit_ibm_runtime import Sampler, Session
import numpy as np
from .preprocessing import preprocessing


class iterative_preprocessing(preprocessing):
    """
        The function initializes the qcl-qpe preprocessing algorithm outlined in [1] with specified parameters.
        
        :param clock: The clock parameter represents the number of bits used to estimate the eigenvalues.
        It is used to calculate the default minimum probability (min_prob) in the
        __init__ method.
        :param backend: The `backend` parameter is used to specify the IBM Backend object that will evaulate
        the circuit.
        :param alpha: The alpha parameter is the initial overestimate of the largest eigenvalue of the system.
        This parameter is not needed if the max_eigenvalue is set.
        :param max_eigenvalue: The `max_eigenvalue` parameter is used to set the maximum eigenvalue for
        the quantum circuit. It determines the maximum value that can be measured for the eigenvalues of
        the observable being measured. If not specified, alpha will be used to determine the maximum eigenvalue 
        via algorithms 1 and 2 from [1].
        :param min_prob: The `min_prob` parameter is used to set the minimum probability value. If
        `min_prob` is not provided, it is set to `2**-clock`, where `clock` is another parameter

        References:
        [1]: Yalovetzky, R., Minssen, P., Herman, D., & Pistoia, M. (2021). 
            NISQ-HHL: Portfolio optimization for near-term quantum hardware. 
            `arXiv:2110.15958 <https://arxiv.org/abs/2110.15958>`_.
        """
    def __init__(self,
                 clock: int,
                 qcl_qpe: bool = False,
                 alpha: float = 50,
                 max_eigenvalue: float = None,
                 min_prob: float = None,
                 result_getter: preprocessing_result_getter = None,
                 **kwargs,
                ):
        self.clock = clock
        self.qcl_qpe = qcl_qpe
        self.alpha = alpha
        self.max_eigenvalue = max_eigenvalue
        if min_prob == None:
           min_prob = 2**-clock
        self.min_prob = min_prob

        if isinstance(result_getter, preprocessing_result_getter):
            self.result_getter = result_getter

        elif result_getter == 'session_result_getter' or 'preprocessing_session_result_getter' or 'session' in kwargs.keys():       
            self.result_getter = preprocessing_session_result_getter(session = kwargs['session'])
        
        elif result_getter == 'backend_result_getter' or 'preprocessing_backend_result_getter' or 'backend' in kwargs.keys():       
            self.result_getter = preprocessing_backend_result_getter(backend = kwargs['backend'])

    
    def get_result(self):
        '''This method runs the QCL_QPE circuit with the specified backend and converts the results from two's complement.'''
        circ = self.construct_circuit(hamiltonian_gate=self.hamiltonian_simulation, state_preparation=self.state_preparation)
        result = self.result_getter.get_result(circ)
        return result
    
    def test_scale(self, scale: float):
        '''This method performs algorithm two from [1].'''
        Gamma = scale/(2**self.clock) # attempt to over approximate the eigenvalue
        self.hamiltonian_simulation = HamiltonianGate(self.problem.A_matrix, -2*np.pi*Gamma)
        results = self.get_result() # get the result
        abs_eigens = {abs(eig) : prob for eig, prob in results.items() if prob > self.min_prob}
        
        if 0 in abs_eigens.keys():
            test = abs_eigens[0] # determine the probability of measureing 0
        else:
            test = 0
        # return a boolean if the eigenvalue is overapproximated
        if test>(1-self.min_prob):
            return True
        else:
            return False
    
    def find_scale(self, alpha: float):
        '''This method combines algorithm 1 and 2 [1]. Combined these algorithms determine the optimal time 
        step parameter of the hamiltonian simulation, if a maximum eigenvalue or hamiltonian gate are not provided.'''
        scale = 1/alpha # initial scale
        self.hamiltonian_simulation = HamiltonianGate(self.problem.A_matrix, -2*np.pi*scale) # set gate
        over_approximation = self.test_scale(scale) # verify over approximation
        while over_approximation == False:
            scale /= 2**(self.clock-1)
            over_approximation = self.test_scale(scale)
        
        x = 0 
        target = int((2**(self.clock-1)-1))
        
        # iteratively adjust scale until the largest eigenvalue is equal to the largest bitstring without overflow
        while x != target:
            self.hamiltonian_simulation = HamiltonianGate(self.problem.A_matrix, -2*np.pi*scale)
            results = self.get_result()
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}  
            x = abs(max(eigens.keys(), key=abs))
            
            if not x == 0:
                scale /= x    
            
            scale *= target
            
        return scale
    
    def adjust_clock(self):
        '''This method determined the minimum number of bits needed to distinguish the lowest eigenvalue of interest from 0,
        which is the required number of bits for eigenvalue inversion'''
        min_eig = None
        # while the minimum eigenvalue is indistinguishable from zero, increase the clock by 1 bit.
        while min_eig == None:
            results = self.get_result(self.scale)
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}
            test = 0
            # if zero is a relevant eigenvalue, increase the clock.
            if 0 in eigens.keys():
                test = eigens[0]
            if test > self.min_prob: 
                self.clock += 1
                self.min_prob /= 2 
            # if the clock is at the set end point, break the loop.
            elif self.clock >= self.max_clock:
                min_eig = 0
            # if the minimum eigenvalue is not zero, set the eigenvalue.
            else:
                min_eig = min(eigens.keys(), key=abs)
    
        return self.clock
    
    def construct_circuit(self, hamiltonian_gate: QuantumCircuit, state_preparation: QuantumCircuit):
        '''Constructs circuit for a given hamiltonian gate'''
        if self.qcl_qpe == False:
            circuit = QuantumCircuit(self.clock+hamiltonian_simulation.num_qubits, self.clock)
            circuit.append(state_preparation, range(self.clock, circuit.num_qubits))
            circuit.append(PhaseEstimation(self.clock, hamiltonian_simulation), circuit.qubits)
            circuit.measure(list(range(self.clock))[::-1], range(self.clock))
            return circuit
        
        else:
            circ = QuantumCircuit(hamiltonian_gate.num_qubits+1, self.clock)
            circ.append(state_preparation, list(range(1, circ.num_qubits)))
            for clbit in range(self.clock):
                if clbit!=0:
                    circ.initialize([1,0],[0])
                circ.h(0)
                power=2**(self.clock-clbit-1)
                ham = hamiltonian_gate.power(power).control()
                circ.append(ham, circ.qubits)
                
                for i in reversed(range(clbit)):
                    
                    if i < self.clock:   
                        N = (2**(i+2))
                        
                        control = clbit-i-1
                        with circ.if_test((control,1)) as passed:
                            circ.p(-np.pi*2/N,0)
                circ.h(0)
                circ.measure(0,[clbit])
            return circ
    
    def estimate(self, problem: QuantumLinearSystemProblem):
        '''Returns a clock-bit estimation of the relevant eigenvalues, and the projection of 
        |b> in the eigenbasis of A.'''
        self.problem=problem

         # If the state_preparation is not specified in the problem, use the standard StatePreparation
        
        if getattr(problem, 'state_preparation', None) is None:
            
            self.state_preparation = StatePreparation(Statevector(problem.b_vector))
        
        else:
            self.state_preparation = problem.state_preparation
        
        # If the hamiltonian simulation is not specified in the problem, use the standard HamiltonianGate
        if getattr(problem, 'hamiltonian_simulation', None) is None:
            if self.max_eigenvalue == None:
                self.scale = self.find_scale(self.alpha)
            else:
                self.scale = abs((0.5-2**-self.clock)/self.max_eigenvalue)
                self.hamiltonian_simulation = HamiltonianGate(problem.A_matrix, -2*np.pi*self.scale)
        
        else:
            self.hamiltonian_simulation = problem.hamiltonian_simulation

        
        if hasattr(self, "max_clock"):
            self.adjust_clock()
            self.scale = abs((0.5-2**-self.clock)/self.max_eigen)
            
        if not hasattr(self, "result"):
            self.get_result(self.scale)
  
        eigenvalue_list = [eig/(self.scale*2**(self.clock)) for eig in self.result.keys() if self.result[eig] > self.min_prob]
        eigenbasis_projection_list = [self.result[eig] for eig in self.result.keys() if self.result[eig] > self.min_prob]
        return eigenvalue_list, eigenbasis_projection_list