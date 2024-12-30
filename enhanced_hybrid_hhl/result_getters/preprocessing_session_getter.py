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
from qiskit_ibm_runtime import SamplerV2, Session
from qiskit import transpile

class preprocessing_session_result_getter(preprocessing_result_getter):
    def __init__(self,
                 session : Session):
        super().__init__()
        self.session = session
        self.backend = session.service.get_backend(session.backend())
        self.sampler = SamplerV2(session)

    def get_result(self,
                   qpe_circ):
        super().get_result()
        '''This method runs the QPE circuit with the ibm_runtime Sampler converts the results from two's complement. '''
        transp = transpile(qpe_circ, self.backend)
        print('circuit depth = ', transp.depth())
        max_key = 2**qpe_circ.num_clbits
        result = self.sampler.run(transp).result()
        
        result_dict = {(key if 2*key<max_key else (key - max_key)) : value for key, value in result.quasi_dists[0].items()}

        return result_dict