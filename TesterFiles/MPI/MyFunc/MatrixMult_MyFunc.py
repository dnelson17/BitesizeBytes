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

if rank == 0:
    #need to update this to match bash script
    core_list = [2**j for j in range(6)]
    time_df = pd.DataFrame(columns=core_list)

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

scatter_time = calc_start - total_start
calc_time = calc_finish - calc_start
gather_time = total_finish - calc_finish
total_time = total_finish - total_start

scatter_sum = np.zeros(0)
calc_sum = np.zeros(0)
gather_sum = np.zeros(0)
total_sum = np.zeros(0)

scatter_sum = comm.reduce(scatter_time, op=MPI.SUM, root=0)
calc_sum = comm.reduce(calc_time, op=MPI.SUM, root=0)
gather_sum = comm.reduce(gather_time, op=MPI.SUM, root=0)
total_sum = comm.reduce(total_time, op=MPI.SUM, root=0)

if rank == 0:
    #Must update this with whatever the max is in the bash file
    scatter_df = pd.read_pickle("scatter_df.pkl")
    calc_df = pd.read_pickle("calc_df.pkl")
    gather_df = pd.read_pickle("gather_df.pkl")
    total_df = pd.read_pickle("total_df.pkl")
    max_cores = 32
    core_list = [2**j for j in range(np.log2(max_cores))]
    if size == 1:
        #add a new line with a new val at the left
        scatter_df = scatter_df.append( pd.DataFrame([(scatter_sum/size) if i==0 else 0.0 for i in range(max_cores)],columns=core_list, index=[max_size]) )
        calc_df = calc_df.append( pd.DataFrame([(calc_sum/size) if i==0 else 0.0 for i in range(max_cores)],columns=core_list, index=[max_size]) )
        gather_df = gather_df.append( pd.DataFrame([(gather_sum/size) if i==0 else 0.0 for i in range(max_cores)],columns=core_list, index=[max_size]) )
        total_df = total_df.append( pd.DataFrame([(total_sum/size) if i==0 else 0.0 for i in range(max_cores)],columns=core_list, index=[max_size]) )
    elif size > 1:
        #add new value at right place
        scatter_df.iloc[mat_size, df.columns.get_loc(str(size))] = (scatter_sum/size)
        calc_df.iloc[mat_size, df.columns.get_loc(str(size))] = (calc_sum/size)
        gather_df.iloc[mat_size, df.columns.get_loc(str(size))] = (gather_sum/size)
        total_df.iloc[mat_size, df.columns.get_loc(str(size))] = (total_sum/size)
    print(scatter_sum/size)
    print(calc_sum/size)
    print(gather_sum/size)
    print(total_sum/size)
    scatter_df.to_pickle("scatter_df.pkl")
    calc_df.to_pickle("calc_df.pkl")
    gather_df.to_pickle("gather_df.pkl")
    total_df.to_pickle("total_df.pkl")

#print(np.array_equal(res, ans))
