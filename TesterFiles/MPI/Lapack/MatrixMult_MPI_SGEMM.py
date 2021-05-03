from scipy.linalg import blas as FB
from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mat_power = int(sys.argv[1])
mat_size = 2**mat_power

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    trans_mat_B = np.transpose(mat_B)
    trans_mat_B = np.ascontiguousarray(trans_mat_B, dtype=np.float32)
else:
    mat_A = None
    trans_mat_B = None

comm.Barrier()
total_start = MPI.Wtime()

power = np.log2(size)/2
pars_i = int(2**(np.ceil(power)))
pars_j = int(2**(np.floor(power)))
len_i = int(mat_size/pars_i)
len_j = int(mat_size/pars_j)
factor = 2**(int(np.log2(size))%2)
count_A = [len_i*mat_size for _ in range(size)]
count_B = [len_j*mat_size for _ in range(size)]
displ_A = [len_i*mat_size * (factor * list_rank // pars_i) for list_rank in range(size)]
displ_B = [len_j*mat_size * (list_rank % pars_j) for list_rank in range(size)]

sub_mat_A = np.zeros((len_i,mat_size),dtype=np.float32)
sub_mat_B = np.zeros((len_j,mat_size),dtype=np.float32)

comm.Scatterv([mat_A,count_A,displ_A,MPI.FLOAT],sub_mat_A,root=0)
comm.Scatterv([trans_mat_B,count_B,displ_B,MPI.FLOAT],sub_mat_B,root=0)

calc_start = MPI.Wtime()

sub_mat_C = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B, trans_b=True)

calc_finish = MPI.Wtime()

if rank == 0:
    mat_C = np.zeros(mat_size*mat_size,dtype=np.float32)
else:
    mat_C = np.zeros(0,dtype=np.float32)

count_C = [len_i*len_j for _ in range(size)]
displ_C = [len_i*len_j*list_rank for list_rank in range(size)]
sub_mat_C = np.ascontiguousarray(sub_mat_C, dtype=np.float32)
comm.Gatherv(sub_mat_C,[mat_C,count_C,displ_C,MPI.FLOAT],root=0)

if rank == 0:
    mat_C = np.split(mat_C, size, axis=0)
    mat_C = np.apply_along_axis(func1d=np.reshape, axis=1, arr=mat_C, newshape=(len_i,len_j) )
    mat_C = np.vstack( np.split( np.concatenate(mat_C,axis=1) , pars_i, axis=1) )

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

