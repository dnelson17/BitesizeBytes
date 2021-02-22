from mpi4py import MPI
import time
import numpy as np
import sys
from scipy.linalg import blas as FB

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI/MPI_IO
# /usr/bin/mpiexec -n 4 python3 MatrixMult.py 4 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode_A = MPI.MODE_RDONLY
amode_B = MPI.MODE_RDONLY
amode_C = MPI.MODE_WRONLY|MPI.MODE_CREATE

#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

#Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
power = np.log2(size)/2
#the number of partitons that must be calculated in the result matrix C in the i and j dimensions
pars_i = int(2**(np.ceil(power)))
pars_j = int(2**(np.floor(power)))
#the size of each partiton in the i and j axis
i_size = int(mat_size/pars_i)
j_size = int(mat_size/pars_j)
#Adjusts partition sizez for odd values of n
factor = 2**(int(np.log2(size))%2)

i_coord = factor * rank // pars_i
j_coord = rank % pars_j

buf_mat_A = np.empty((i_size,mat_size), dtype=np.float32)
buf_mat_B = np.empty((j_size,mat_size), dtype=np.float32)

offset_A = i_coord*buf_mat_A.nbytes
offset_B = j_coord*buf_mat_B.nbytes

t_start = MPI.Wtime()

fh_A = MPI.File.Open(comm, f"mat_A/mat_A_{mat_size}_{iteration}.txt", amode_A)
fh_B = MPI.File.Open(comm, f"mat_B/mat_B_{mat_size}_{iteration}.txt", amode_B)
    
fh_A.Read_at_all(offset_A, buf_mat_A)
fh_B.Read_at_all(offset_B, buf_mat_B)

buf_mat_C = FB.sgemm(alpha=1.0, a=buf_mat_A, b=buf_mat_B)

fh_C = MPI.File.Open(comm, f"mat_C/mat_C_{mat_size}_{iteration}.txt", amode_C)

#need to add code to write results to mat_C file
offset_C = 

fh_C.Write_at_all(offset_C, buf_mat_C)

t_diff = MPI.Wtime() - t_start
print(f"rank: {rank}, time: {t_diff}")

fh_A.Close()
fh_B.Close()
fh_C.Close()
