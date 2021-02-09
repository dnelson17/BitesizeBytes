from mpi4py import MPI
import time
import numpy as np
import sys

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI/MPI_IO
# /usr/bin/mpiexec -n 4 python3 MatrixMultNumpy.py 8 8

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
                mat_C[i,j] += mat_A[i][k] * mat_B[j][k]
    return mat_C


numberRows = int( sys.argv[1])
numberColumns = int( sys.argv[2])

assert numberRows == numberColumns

mat_size = numberRows

power = np.log2(size)/2
#represents the number of partitons that must be calculated in the result matrix C
i_len = int(2**(np.ceil(power)))
j_len = int(2**(np.floor(power)))
i_size = mat_size/i_len
j_size = mat_size/j_len

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    ans = np.matmul(mat_A,mat_B)
    
    t_start = MPI.Wtime()

    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([int(i*i_size),int(j*j_size)])
    print(send_list)
else:
    send_list = None

comm.barrier()

info = comm.scatter(send_list,root=0)

print(f"rank: {rank}, info[0]: {info[0]}, i_size: {i_size}, info[1]: {info[1]}, j_size: {j_size}")
#mat_A = np.loadtxt("mat_A.txt",skiprows=info[0],max_rows=i_size)
#mat_B = np.loadtxt("mat_B.txt",skiprows=info[1],max_rows=j_size)
mat_A = np.loadtxt("mat_A.txt",skiprows=4,max_rows=4)
mat_B = np.loadtxt("mat_B.txt",skiprows=4,max_rows=4)
mat_C = matrix_mult(mat_A,mat_B)

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
    np.savetxt("mat_C.txt")
    t_diff = MPI.Wtime() - t_start
    print(t_diff)
    #print(np.array_equal(res, ans))
    
