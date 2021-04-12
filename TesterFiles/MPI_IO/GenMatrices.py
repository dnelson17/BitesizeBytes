from mpi4py import MPI
import numpy as np
import sys

# mpiexec -n 1 python GenMatrices.py 4 1

comm = MPI.COMM_WORLD

# Will use this when testing such that it can be looped efficiently
mat_power = int(sys.argv[1])
iteration = int(sys.argv[2])
mat_size = 2**mat_power

mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
#print("mat A")
#print(mat_A)
#As B is an arbitrary matrix, we will assume it has already been transposed
t_mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
mat_B = np.transpose(t_mat_B)[:]
#print("mat B (pre transpose)")
#print(mat_B)
#print("mat B (post transpose)")
#print(t_mat_B)


amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

fh_A = MPI.File.Open(comm, f"mat_A/mat_A_{mat_size}_{iteration}.txt", amode)
fh_A.Write_at_all(0,mat_A)
fh_A.Close()

fh_B = MPI.File.Open(comm, f"mat_B/mat_B_{mat_size}_{iteration}.txt", amode)
fh_B.Write_at_all(0,t_mat_B)
fh_B.Close()

#print("Mat C")
#mat_C_res = np.matmul(mat_A,mat_B)
#print(mat_C_res)

