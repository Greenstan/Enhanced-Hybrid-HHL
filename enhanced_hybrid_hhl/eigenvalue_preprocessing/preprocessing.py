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

from abc import ABC, abstractmethod
from ..quantum_linear_system import QuantumLinearSystemProblem

class preprocessing(ABC):
    @abstractmethod
    def estimate(self,
                 problem : QuantumLinearSystemProblem) -> tuple[list, list]:
        """
        Abstract method to estimate eigenvalues and relvance for a Quantum Linear System Problem.

        Parameters:
        - problem: The QuantumLinearSystemProblem to solve (type: any).

        Returns:
        - A tuple of two lists (eigenvalue_list, eigenbasis_projection_list).
        """
        pass