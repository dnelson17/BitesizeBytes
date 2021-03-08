from mpi4py import MPI
import time
import numpy as np
import sys

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI
# /usr/bin/mpiexec -n 4 python3 scatter.py 4 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
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
    ans = np.matmul(mat_A,mat_B)
    
    t_start = MPI.Wtime()
    
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    send_list_A = np.split(mat_A, i_len, axis=0)
    send_list_B = np.split(mat_B, j_len, axis=1)
    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([send_list_A[i],send_list_B[j]])
else:
    mat_A = None
    mat_B = None
    send_list = None

mats = comm.scatter(send_list,root=0)

mat_C = matrix_mult(mats[0],mats[1])

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
    t_diff = MPI.Wtime() - t_start
    print(t_diff)
    #print(np.array_equal(res, ans))
    
