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
from qiskit import transpile

class preprocessing_backend_result_getter(preprocessing_result_getter):
        '''This method runs the QCL_QPE circuit with the specified backend and converts the results from two's complement.'''
        def __init__(self,
                     backend: Backend,
                     wait_for_result: bool = True,
                     noise_model: str = None,
                     shots=4000):
                self.backend = backend
                self.wait_for_result = wait_for_result
                self.noise_model = noise_model
                self.shots = shots
        
        def get_result(self,
                       qpe_circ):
            super().get_result()

            transp = transpile(qpe_circ, self.backend)
            if self.noise_model is None:   
                job = self.backend.run(transp, shots=4000)

            else: 
                job = self.backend.run(transp, noise_model=self.noise_model, shots=4000)
            
            if self.wait_for_result:      
                result = job.result()
                counts = result.get_counts()
                tot = sum(counts.values())
                # translate results into integer representation of the bitstring and adjust for two's compliment
                result_dict = {(int(key,2) if key[0]=='0' else (int(key,2) - (2**(len(key))))) : value / tot for key, value in counts.items()}
                return result_dict
            
            else:
                 return job