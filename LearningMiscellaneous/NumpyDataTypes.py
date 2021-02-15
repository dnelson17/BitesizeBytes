#Checking whether the runtime of LAPACK SGEMM will be drastically changed
#if the NUmpy arrays are type casted as single point precision when being defined

import numpy as np
import time
from scipy.linalg import blas as FB

mat_size = 8192

mat_A = np.random.rand(mat_size,mat_size)
mat_B = np.random.rand(mat_size,mat_size)

t0 = time.time()

mat_C = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)

t1 = time.time()

mat_A_new = mat_A.astype(np.float32)
mat_B_new = mat_B.astype(np.float32)

t2 = time.time()

mat_C_new = FB.sgemm(alpha=1.0, a=mat_A_new, b=mat_B_new)

t3 = time.time()


print(t1-t0)
print(t3-t2)
