import numpy as np
from scipy.linalg import blas as FB

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[0]))
    print(mat_C.shape)
    for i in range(len(mat_A)):
        for j in range(len(mat_B)):
            for k in range(len(mat_B[i])):
                mat_C[i,j] += mat_A[i][k] * mat_B[j][k]
    return mat_C


mat_A = np.loadtxt("mat_A.txt")
mat_B = np.loadtxt("mat_B.txt")

#answer = np.matmul(mat_A,mat_B)
#answer = matrix_mult(mat_A,mat_B)
answer = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)


print(answer)

res = np.loadtxt("mat_C.txt")

print(res)

print(np.array_equal(answer,res))
