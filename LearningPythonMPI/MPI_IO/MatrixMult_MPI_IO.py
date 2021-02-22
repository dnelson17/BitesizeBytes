from mpi4py import MPI
import time
import numpy as np
import sys
from scipy.linalg import blas as FB

# cd BitesizeBytes/LearningPythonMPI/MPI_IO
# /usr/bin/mpiexec -n 4 python3 MatrixMult_MPI_IO.py 4 4

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
#Adjusts partition sizes for odd values of n
factor = 2**(int(np.log2(size))%2)
#Calculates to coordinates of the result block matrix in mat C
i_coord = factor * rank // pars_i
j_coord = rank % pars_j

t_start = MPI.Wtime()

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

calc_start = MPI.Wtime()

buf_mat_C = FB.sgemm(alpha=1.0, a=buf_mat_A, b=mat_B)

calc_time = MPI.Wtime() - calc_start
print(calc_time)

fh_C = MPI.File.Open(comm, f"mat_C/mat_C_{mat_size}_{iteration}.txt", amode_C)
filetype = MPI.FLOAT.Create_vector(j_size, i_size, mat_size)
filetype.Commit()
offset_C = (mat_size*i_coord*i_size + j_coord*j_size)*MPI.FLOAT.Get_size()
fh_C.Set_view(offset_C, filetype=filetype)
fh_C.Write_all(buf_mat_C)
filetype.Free()
fh_C.Close()

total_time = MPI.Wtime() - t_start

comm.Barrier()
print(total_time)

calc_sum = np.zeros(0)
total_sum = np.zeros(0)

calc_sum = comm.reduce(calc_time, op=MPI.SUM, root=0)
total_sum = comm.reduce(total_time, op=MPI.SUM, root=0)

if rank == 0:
    print(f"avg calc_time: {calc_sum/size}")
    print(f"avg total_time: {total_sum/size}")
