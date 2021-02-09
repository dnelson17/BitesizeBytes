import numpy as np
from scipy.linalg import blas as FB
import sys

mat_A = np.loadtxt("mat_A.txt")
mat_B = np.loadtxt("mat_B.txt")
mat_B = np.transpose(mat_B)

answer = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)

res = np.loadtxt("mat_C.txt")

print(np.allclose(answer,res))
