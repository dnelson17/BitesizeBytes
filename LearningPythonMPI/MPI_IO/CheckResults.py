import numpy as np

mat_A = np.loadtxt("mat_A.txt")
mat_B = np.loadtxt("mat_B.txt")

mat_C = np.matmul(mat_A,mat_B)

