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
mat_A = None
mat_B = None
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.transpose(mat_B)
    mat_B = np.ascontiguousarray(mat_B, dtype=np.float32)

comm.Barrier()
total_start = MPI.Wtime()

power = np.log2(size)/2
pars_i = int(2**(np.ceil(power)))
pars_j = int(2**(np.floor(power)))
len_i = int(mat_size/pars_i)
len_j = int(mat_size/pars_j)
factor = 2**(int(np.log2(size))%2)
displ_A = [len_i * (factor * list_rank // pars_i) for list_rank in range(size)]
displ_B = [len_j * (list_rank % pars_j) for list_rank in range(size)]

sub_mat_A = np.empty((len_i,mat_size),dtype=np.float32)
sub_mat_B = np.empty((len_j,mat_size),dtype=np.float32)

if rank == 0:
    for i in range(1,size):
        comm.Send([mat_A[displ_A[i]:displ_A[i]+len_i],MPI.FLOAT],dest=i,tag=25)
        comm.Send([mat_B[displ_B[i]:displ_B[i]+len_j],MPI.FLOAT],dest=i,tag=25)
    sub_mat_A = mat_A[displ_A[0]:displ_A[0]+len_i]
    sub_mat_B = mat_B[displ_B[0]:displ_B[0]+len_j]
else:
    comm.Recv([sub_mat_A,MPI.FLOAT],source=0)
    comm.Recv([sub_mat_B,MPI.FLOAT],source=0)

del mat_A
del mat_B 

comm.Barrier()
calc_start = MPI.Wtime()

sub_mat_C = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B, trans_b=True)

comm.Barrier()
calc_finish = MPI.Wtime()

del sub_mat_A
del sub_mat_B

mat_C = None
if rank == 0:
    mat_C = np.empty(mat_size*mat_size,dtype=np.float32)

count_C = [len_i*len_j for _ in range(size)]
displ_C = [len_i*len_j*list_rank for list_rank in range(size)]
sub_mat_C = np.ascontiguousarray(sub_mat_C, dtype=np.float32)
comm.Gatherv(sub_mat_C,[mat_C,count_C,displ_C,MPI.FLOAT],root=0)

if rank == 0:
    mat_C = np.split(mat_C, size, axis=0)
    mat_C = np.apply_along_axis(func1d=np.reshape, axis=1, arr=mat_C, newshape=(len_i,len_j) )
    mat_C = np.vstack( np.split( np.concatenate(mat_C,axis=1) , pars_i, axis=1) )

comm.Barrier()
total_finish = MPI.Wtime()

#   The matrix mult is now done, everything past this is just saving timing results.

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
    max_core_power = 6
    core_list = [2**j for j in range(max_core_power)]
    if size == 1:
        #add a new line with a new val at the left
        scatter_df = scatter_df.append( pd.DataFrame([[scatter_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        calc_df = calc_df.append( pd.DataFrame([[calc_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        gather_df = gather_df.append( pd.DataFrame([[gather_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        total_df = total_df.append( pd.DataFrame([[total_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
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

