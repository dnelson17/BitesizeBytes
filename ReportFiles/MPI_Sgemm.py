from scipy.linalg import blas as FB
from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Reads in the order of the matrix that will be worked on from the command line
mat_power = int(sys.argv[1])
mat_size = 2**mat_power

# Initialize the 2 random matrices A and B only if this is rank 0, the master core
mat_A = None
mat_B = None
if rank == 0:
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    #Transpose B so that it is easier to send to workers
    mat_B = np.transpose(mat_B)
    mat_B = np.ascontiguousarray(mat_B, dtype=np.float32)

#Timing starts for the "scatter" portion
comm.Barrier()
total_start = MPI.Wtime()

#Calclates x for number of processors P=2^x
power = np.log2(size)/2
#Calclulates the number of partitions in the i,j axes, phi_i and phi_j, respectively
pars_i = int(2**(np.ceil(power)))
pars_j = int(2**(np.floor(power)))
#Calclulates the length of partitions in the i,j axes, psi_i and psi_j, respectively
len_i = int(mat_size/pars_i)
len_j = int(mat_size/pars_j)
#Determines whether x is even or odd for P=2^x
factor = 2**(int(np.log2(size))%2)
#Creates a list of the coordinates that each core will be working on
displ_A = [len_i * (factor * list_rank // pars_i) for list_rank in range(size)]
displ_B = [len_j * (list_rank % pars_j) for list_rank in range(size)]

#Creates empty matrices for each worker's submatrices of A,B to be sent to
sub_mat_A = np.empty((len_i,mat_size),dtype=np.float32)
sub_mat_B = np.empty((len_j,mat_size),dtype=np.float32)

#The master core will iterate over every worker and send them their respective submatrices
if rank == 0:
    for i in range(1,size):
        comm.Send([mat_A[displ_A[i]:displ_A[i]+len_i],MPI.FLOAT],dest=i,tag=25)
        comm.Send([mat_B[displ_B[i]:displ_B[i]+len_j],MPI.FLOAT],dest=i,tag=25)
    #Defines the matrices that the master core will be operating on
    sub_mat_A = mat_A[displ_A[0]:displ_A[0]+len_i]
    sub_mat_B = mat_B[displ_B[0]:displ_B[0]+len_j]
else:
    #Every worker core receives their submatrices of A,B
    comm.Recv([sub_mat_A,MPI.FLOAT],source=0)
    comm.Recv([sub_mat_B,MPI.FLOAT],source=0)

#Starts the timer for the beginning of the "calculation" portion
comm.Barrier()
calc_start = MPI.Wtime()

#Each core calculates their submatrix C' using sgemm. In the handwritten version of this, the "matrix_mult" function will be called instead
sub_mat_C = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B, trans_b=True)

#Stops the timer for the "calculation" portion, starting the timer for the "gather" portion
comm.Barrier()
calc_finish = MPI.Wtime()

#Creates an empty matrix for the submatrices C' to be gathered into
mat_C = None
if rank == 0:
    mat_C = np.empty(mat_size*mat_size,dtype=np.float32)

#Gathers all of the submatrices C'
count_C = [len_i*len_j for _ in range(size)]
displ_C = [len_i*len_j*list_rank for list_rank in range(size)]
sub_mat_C = np.ascontiguousarray(sub_mat_C, dtype=np.float32)
comm.Gatherv(sub_mat_C,[mat_C,count_C,displ_C,MPI.FLOAT],root=0)

#Restructures all of the submatrices into the final result matrix C
if rank == 0:
    mat_C = np.split(mat_C, size, axis=0)
    mat_C = np.apply_along_axis(func1d=np.reshape, axis=1, arr=mat_C, newshape=(len_i,len_j) )
    mat_C = np.vstack( np.split( np.concatenate(mat_C,axis=1) , pars_i, axis=1) )

#Stops the timer as the matrix multiplication is now finished
comm.Barrier()
total_finish = MPI.Wtime()

#The matrix multiplication is now done, everything past this is just saving timing results.

