import numpy as np
import sys

# Will use this when testing such that it can be looped efficiently
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
#As B is an arbitrary matrix, we will take the transpose of it
mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
mat_B = np.transpose(mat_B)

np.savetxt(f"mat_A/mat_A_{mat_size}_{iteration}.txt", mat_A)
np.savetxt(f"mat_B/mat_B_{mat_size}_{iteration}.txt", mat_B)
