from mpi4py import MPI
import pandas as pd
import numpy as np
import time
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


mat_power = int(sys.argv[1])
mat_size = 2**mat_power

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    #ans = np.matmul(mat_A,mat_B)

total_start = MPI.Wtime()

if rank == 0:
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

mat_C = matrix_mult(mats[0],mats[1])

calc_finish = MPI.Wtime()

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , pars_i, axis=1) )

total_finish = MPI.Wtime()

proc0_total_start = comm.bcast(total_start,root=0)

time_difference = proc0_total_start - total_start

total_start += time_difference
calc_start += time_difference
calc_finish += time_difference
total_finish += time_difference

#print(f"rank: {rank}, total_start: {total_start}, calc_start: {calc_start}, calc_finish: {calc_finish}, total_finish: {total_finish}, time_difference: {time_difference}")

total_start_min = comm.reduce(total_start, op=MPI.MIN, root=0)
calc_start_min = comm.reduce(calc_start, op=MPI.MIN, root=0)
calc_finish_max = comm.reduce(calc_finish, op=MPI.MAX, root=0)
total_finish_max = comm.reduce(total_finish, op=MPI.MAX, root=0)

if rank == 0:
    scatter_time = calc_start_min - total_start_min
    calc_time = calc_finish_max - calc_start_min
    gather_time = total_finish_max - calc_finish_max
    total_time = total_finish_max - total_start_min
    assert np.isclose(scatter_time+calc_time+gather_time,total_time)
    #Must update this with whatever the max is in the bash file
    scatter_df = pd.read_pickle("Time_dfs/scatter_df.pkl")
    calc_df = pd.read_pickle("Time_dfs/calc_df.pkl")
    gather_df = pd.read_pickle("Time_dfs/gather_df.pkl")
    total_df = pd.read_pickle("Time_dfs/total_df.pkl")
    max_cores = 32
    core_list = [2**j for j in range(int(np.log2(max_cores))+1)]
    if size == 1:
        #add a new line with a new val at the left
        scatter_df = scatter_df.append( pd.DataFrame([[scatter_time if i==0 else 0.0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[mat_size]) )
        calc_df = calc_df.append( pd.DataFrame([[calc_time if i==0 else 0.0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[mat_size]) )
        gather_df = gather_df.append( pd.DataFrame([[gather_time if i==0 else 0.0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[mat_size]) )
        total_df = total_df.append( pd.DataFrame([[total_time if i==0 else 0.0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[mat_size]) )
    elif size > 1:
        #add new value at right place
        size_power = int(np.log2(size))
        scatter_df.iloc[-1, size_power] = scatter_time
        calc_df.iloc[-1, size_power] = calc_time
        gather_df.iloc[-1, size_power] = gather_time
        total_df.iloc[-1, size_power] = total_time
    print(f"scatter: {scatter_time}")
    print(f"calc: {calc_time}")
    print(f"gather: {gather_time}")
    print(f"total: {total_time}")
    scatter_df.to_pickle("Time_dfs/scatter_df.pkl")
    calc_df.to_pickle("Time_dfs/calc_df.pkl")
    gather_df.to_pickle("Time_dfs/gather_df.pkl")
    total_df.to_pickle("Time_dfs/total_df.pkl")
