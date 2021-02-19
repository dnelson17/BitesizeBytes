from mpi4py import MPI
import time
import numpy as np
import sys
from scipy.linalg import blas as FB

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI/MPI_IO
# /usr/bin/mpiexec -n 4 python3 MatrixMultSGEMM.py 8

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

#Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
power = np.log2(size)/2
#Represents the number of partitons that must be calculated in the result matrix C
i_len = int(2**(np.ceil(power)))
j_len = int(2**(np.floor(power)))
#Represents the size of each partiton in the i and j axis
i_size = int(mat_size/i_len)
j_size = int(mat_size/j_len)

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    t_start = MPI.Wtime()
    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([int(i*i_size),int(j*j_size)])
else:
    send_list = None

info = comm.scatter(send_list,root=0)

mat_A = np.loadtxt(f"mat_A/mat_A_{mat_size}_{iteration}.txt",skiprows=info[0],max_rows=i_size)
mat_B = np.loadtxt(f"mat_B/mat_B_{mat_size}_{iteration}.txt",skiprows=info[1],max_rows=j_size)
mat_B = np.transpose(mat_B)
mat_C = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
    np.savetxt(f"mat_C/mat_C_{mat_size}_{iteration}.txt",res)
    t_diff = MPI.Wtime() - t_start
    print(t_diff)
    
