from mpi4py import MPI
import time
import numpy as np
import sys
from scipy.linalg import blas as FB

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI
# /usr/bin/mpiexec -n 4 python3 MatrixMult_SGEMM.py 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mat_size = int(sys.argv[1])

total_start = MPI.Wtime()

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    #ans = np.matmul(mat_A,mat_B)
    
    power = np.log2(size)/2
    pars_i = int(2**(np.ceil(power)))
    pars_j = int(2**(np.floor(power)))
    send_list_A = np.split(mat_A, pars_i, axis=0)
    send_list_B = np.split(mat_B, pars_j, axis=1)
    send_list = []
    for i in range(pars_i):
        for j in range(pars_j):
            send_list.append([send_list_A[i],send_list_B[j]])
    #mat_A = None
    #mat_B = None
else:
    mat_A = None
    mat_B = None
    send_list = None

mats = comm.scatter(send_list,root=0)

calc_start = MPI.Wtime()

mat_C = FB.sgemm(alpha=1.0, a=mats[0], b=mats[1])

calc_finish = MPI.Wtime()

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , pars_i, axis=1) )

total_finish = MPI.Wtime()

scatter_time = calc_start - total_start
calc_time = calc_finish - calc_start
gather_time = total_finish - calc_finish

scatter_sum = np.zeros(0)
calc_sum = np.zeros(0)
gather_sum = np.zeros(0)

scatter_sum = comm.reduce(scatter_time, op=MPI.SUM, root=0)
calc_sum = comm.reduce(calc_time, op=MPI.SUM, root=0)
gather_sum = comm.reduce(gather_time, op=MPI.SUM, root=0)

if rank == 0:
    print(scatter_sum/size)
    print(calc_sum/size)
    print(gather_sum/size)

#print(np.array_equal(res, ans))
