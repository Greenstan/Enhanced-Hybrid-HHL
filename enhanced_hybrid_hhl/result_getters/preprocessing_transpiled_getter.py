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

from .preprocessing_result_getter import preprocessing_result_getter
from qiskit.providers import Backend
from qiskit.transpiler import PassManager
from qiskit import transpile

class preprocessing_transpiled_result_getter(preprocessing_result_getter):
        '''This method runs the QCL_QPE circuit with the specified backend and converts the results from two's complement.'''
        def __init__(self,
                     backend: Backend,
                     pass_manager: PassManager = None,
                     ):
                self.backend = backend
                self.pass_manager = pass_manager
                     
        def get_result(self,
                       qpe_circ):
            super().get_result()

            if self.pass_manager is None:
                circuit = transpile(qpe_circ, backend=self.backend)
            
            else:
                circuit = self.pass_manager.run(qpe_circ)

            return circuit