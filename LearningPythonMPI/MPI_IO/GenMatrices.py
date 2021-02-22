import numpy as np
import sys

comm = MPI.COMM_WORLD

# Will use this when testing such that it can be looped efficiently
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
#As B is an arbitrary matrix, we will take the transpose of it
mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
mat_B = np.transpose(mat_B)

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

fh_A = MPI.File.Open(comm, f"mat_A/mat_A_{mat_size}_{iteration}.txt", amode)
fh_A.Write_at_all(0,mat_A)
fh_A.Close()

fh_B = MPI.File.Open(comm, f"mat_B/mat_B_{mat_size}_{iteration}.txt", amode)
fh_B.Write_at_all(0,mat_B)
fh_B.Close()
