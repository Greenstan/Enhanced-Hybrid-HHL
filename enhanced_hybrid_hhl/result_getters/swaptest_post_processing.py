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
import numpy as np

def st_post_processing(result):

    # IonQ result
    if '0 1' in result.keys():
        counts_01 = result['0 1']
        if '1 1' in result.keys():
            counts_11 = result['1 1']
        else:
            counts_11 = 0

    # IBM and IQM result
    else:
        counts_01 = result['1']
        counts_11 = result['3']
    if counts_01 <= counts_11:
        return 0
    else:
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)