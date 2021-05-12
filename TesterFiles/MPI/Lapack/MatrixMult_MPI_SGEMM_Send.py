from scipy.linalg import blas as FB
from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys

# mpiexec -n 32 python MatrixMult_MPI_SGEMM_Send.py 13


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mat_power = int(sys.argv[1])
mat_size = 2**mat_power

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.transpose(mat_B)
    mat_B = np.ascontiguousarray(mat_B, dtype=np.float32)
else:
    mat_A = None
    mat_B = None

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

mat_A = None
mat_B = None

comm.Barrier()
calc_start = MPI.Wtime()

sub_mat_C = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B, trans_b=True)

comm.Barrier()
calc_finish = MPI.Wtime()

sub_mat_A = None
sub_mat_B = None

mat_C = None
if rank == 0:
    mat_C = np.empty(mat_size*mat_size,dtype=np.float32)

count_C = [len_i*len_j for _ in range(size)]
displ_C = [len_i*len_j*list_rank for list_rank in range(size)]
sub_mat_C = np.ascontiguousarray(sub_mat_C, dtype=np.float32)
comm.Gatherv(sub_mat_C,[mat_C,count_C,displ_C,MPI.FLOAT],root=0)

comm.Barrier()
total_finish = MPI.Wtime()

if rank == 0:
    mat_C = np.split(mat_C, size, axis=0)
    mat_C = np.apply_along_axis(func1d=np.reshape, axis=1, arr=mat_C, newshape=(len_i,len_j) )
    mat_C = np.vstack( np.split( np.concatenate(mat_C,axis=1) , pars_i, axis=1) )

