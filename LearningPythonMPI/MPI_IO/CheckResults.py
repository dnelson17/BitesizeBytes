import numpy as np
from scipy.linalg import blas as FB
import sys

#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
size = int(sys.argv[2])
iteration = int(sys.argv[3])

mat_A = np.loadtxt(f"mat_A_{mat_size}_{size}_{iteration}.txt")
mat_B = np.loadtxt(f"mat_B_{mat_size}_{size}_{iteration}.txt")
mat_B = np.transpose(mat_B)

answer = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)

res = np.loadtxt(f"mat_C_{mat_size}_{size}_{iteration}.txt")

print(np.allclose(answer,res))
