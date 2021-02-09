import numpy as np
import sys

# Will use this when testing such that it can be looped efficiently
mat_size = int( sys.argv[1])
size = int( sys.argv[2])
iteration = int( sys.argv[3])

mat_A = np.random.rand(mat_size,mat_size)
#As B is an arbitrary matrix, we will take the transpose of it
mat_B = np.random.rand(mat_size,mat_size)
mat_B = np.transpose(mat_B)

np.savetxt(f"mat_A_{mat_size}_{no_cores}_{iteration}.txt", mat_A)
np.savetxt(f"mat_B_{mat_size}_{no_cores}_{iteration}.txt", mat_B)
