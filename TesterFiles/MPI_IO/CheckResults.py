from mpi4py import MPI
import numpy as np
from scipy.linalg import blas as FB
import sys

# mpiexec -n 1 python CheckResults.py 4 1

comm = MPI.COMM_WORLD


#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

amode = MPI.MODE_RDONLY

#Opening and reading matrix A
fh_A = MPI.File.Open(comm, f"mat_A/mat_A_{mat_size}_{iteration}.txt", amode)
buf_mat_A = np.empty((mat_size,mat_size), dtype=np.float32)
fh_A.Read_at_all(0, buf_mat_A)
fh_A.Close()
#Opening and reading matrix B
fh_B = MPI.File.Open(comm, f"mat_B/mat_B_{mat_size}_{iteration}.txt", amode)
buf_mat_B = np.empty((mat_size,mat_size), dtype=np.float32)
fh_B.Read_at_all(0, buf_mat_B)
fh_B.Close()

#Opening and reading matrix C
fh_C = MPI.File.Open(comm, f"mat_C/mat_C_{mat_size}_{iteration}.txt", amode)
buf_mat_C = np.empty((mat_size,mat_size), dtype=np.float32)
fh_C.Read_at_all(0, buf_mat_C)
fh_C.Close()

answer = np.matmul(buf_mat_A,np.transpose(buf_mat_B))

#print("mat A")
#print(buf_mat_A)
#print("mat B")
#print(buf_mat_B)
#print("My result")
#print(buf_mat_C)

#print("Real result")
#print(answer)

print(np.allclose(answer,buf_mat_C))
