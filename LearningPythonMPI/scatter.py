from mpi4py import MPI
import time
import numpy as np
from scipy.linalg import blas as FB
import sys 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def matrix_mult(first_row, last_row, mat_A, mat_B):
    mat_C = np.zeros(mat_B.shape)
    for i in range(first_row, last_row):
        for j in range(mat_size):
            for k in range(mat_size):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    return mat_C


numberRows = int( sys.argv[1])
numberColumns = int( sys.argv[2])

assert numberRows == numberColumns

mat_size = numberRows

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    #need to split the matrices here s.t. they can be scattered
    for i in range(size):
        first_i = round(i * mat_size / size)
        last_i = round((i + 1) * mat_size / size)
        for j in range(size):
            first_j = round(j * mat_size / size)
            last_j = round((j + 1) * mat_size / size)
else:
    mat_A = None
    mat_B = None

mat_A = comm.scatter(mat_A,root=0)
mat_B = comm.scatter(mat_B,root=0)

mat_C = matrix_mult(mat_A,mat_B)

final_mat_C = comm.gather(mat_C,root=0)



# Scatter matrices to all processes
print("Process ", rank, " before n = ", n[0])
comm.Bcast(n, root=0)
print("Process ", rank, " after n = ", n[0])

# Compute partition
h = (b - a) / (n * size) # calculate h *after* we receive n
a_i = a + rank * h * n
my_int[0] = integral(a_i, h, n[0])

# Send partition back to root process, computing sum across all partitions
print("Process ", rank, " has the partial integral ", my_int[0])
comm.Reduce(my_int, integral_sum, MPI.SUM, dest)

# Only print the result in process 0
if rank == 0:
    print('The Integral Sum =', integral_sum[0])
