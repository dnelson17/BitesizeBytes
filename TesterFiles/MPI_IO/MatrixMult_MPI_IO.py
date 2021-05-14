from scipy.linalg import blas as FB
from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode_A = MPI.MODE_RDONLY
amode_B = MPI.MODE_RDONLY
amode_C = MPI.MODE_WRONLY|MPI.MODE_CREATE

#Reads the Matrix Size from the command line
mat_power = int(sys.argv[1])
iteration = int(sys.argv[2])
mat_size = 2**mat_power

#Assuming the number of processors is of size 2^n for int n, we take log2 to find the value of n
power = np.log2(size)/2
#the number of partitons that must be calculated in the result matrix C in the i and j dimensions
pars_i = int(2**(np.ceil(power)))
pars_j = int(2**(np.floor(power)))
#the size of each partiton in the i and j axis
i_size = int(mat_size/pars_i)
j_size = int(mat_size/pars_j)
#Adjusts partition sizes for odd values of n
factor = 2**(int(np.log2(size))%2)
#Calculates to coordinates of the result block matrix in mat C
i_coord = factor * rank // pars_i
j_coord = rank % pars_j

comm.Barrier()
io_start = MPI.Wtime()

#Opening and reading matrix A
fh_A = MPI.File.Open(comm, f"mat_A/mat_A_{mat_size}_{iteration}.txt", amode_A)
buf_mat_A = np.empty((i_size,mat_size), dtype=np.float32)
offset_A = i_coord*buf_mat_A.nbytes
fh_A.Read_at_all(offset_A, buf_mat_A)
fh_A.Close()
#Opening and reading matrix B
fh_B = MPI.File.Open(comm, f"mat_B/mat_B_{mat_size}_{iteration}.txt", amode_B)
buf_mat_B = np.empty((j_size,mat_size), dtype=np.float32)
offset_B = j_coord*buf_mat_B.nbytes
fh_B.Read_at_all(offset_B, buf_mat_B)
mat_B = np.transpose(buf_mat_B)
fh_B.Close()

comm.Barrier()
calc_start = MPI.Wtime()

mat_C = FB.sgemm(alpha=1.0, a=buf_mat_A, b=mat_B)

comm.Barrier()
calc_finish = MPI.Wtime()

buf_mat_C = np.ascontiguousarray(mat_C)

fh_C = MPI.File.Open(comm, f"mat_C/mat_C_{mat_size}_{iteration}.txt", amode_C)
filetype = MPI.FLOAT.Create_vector(i_size, j_size, mat_size)
filetype.Commit()
offset_C = (mat_size*i_coord*i_size + j_coord*j_size)*MPI.FLOAT.Get_size()
fh_C.Set_view(offset_C, filetype=filetype)
fh_C.Write_all(buf_mat_C)
filetype.Free()
fh_C.Close()

comm.Barrier()
io_finish = MPI.Wtime()


proc0_total_start = comm.bcast(io_start,root=0)

time_difference = proc0_total_start - io_start

io_start += time_difference
calc_start += time_difference
calc_finish += time_difference
io_finish += time_difference

io_start = comm.reduce(io_start, op=MPI.SUM, root=0)
calc_start = comm.reduce(calc_start, op=MPI.SUM, root=0)
calc_finish = comm.reduce(calc_finish, op=MPI.SUM, root=0)
io_finish = comm.reduce(io_finish, op=MPI.SUM, root=0)

if rank == 0:
    io_start /= size
    calc_start /= size
    calc_finish /= size
    io_finish /= size
    read_time = calc_start - io_start
    calc_time = calc_finish - calc_start
    write_time = io_finish - calc_finish
    total_time = io_finish - io_start
    assert np.isclose(read_time+calc_time+write_time,total_time)
    #Must update this with whatever the max is in the bash file
    read_df = pd.read_pickle("Time_dfs/read_df.pkl")
    calc_df = pd.read_pickle("Time_dfs/calc_df.pkl")
    write_df = pd.read_pickle("Time_dfs/write_df.pkl")
    total_df = pd.read_pickle("Time_dfs/total_df.pkl")
    max_cores = 32
    max_core_power = 6
    core_list = [2**j for j in range(max_core_power)]
    if size == 1:
        #add a new line with a new val at the left
        read_df = read_df.append( pd.DataFrame([[read_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        calc_df = calc_df.append( pd.DataFrame([[calc_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        write_df = write_df.append( pd.DataFrame([[write_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
        total_df = total_df.append( pd.DataFrame([[total_time if i==0 else 0.0 for i in range(max_core_power)]],columns=core_list, index=[mat_size]) )
    elif size > 1:
        #add new value at right place
        size_power = int(np.log2(size))
        read_df.iloc[-1, size_power] = read_time
        calc_df.iloc[-1, size_power] = calc_time
        write_df.iloc[-1, size_power] = write_time
        total_df.iloc[-1, size_power] = total_time
    print(f"read: {read_time}")
    print(f"calc: {calc_time}")
    print(f"write: {write_time}")
    print(f"total: {total_time}")
    read_df.to_pickle("Time_dfs/read_df.pkl")
    calc_df.to_pickle("Time_dfs/calc_df.pkl")
    write_df.to_pickle("Time_dfs/write_df.pkl")
    total_df.to_pickle("Time_dfs/total_df.pkl")
