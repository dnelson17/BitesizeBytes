import numpy as np
import sys

mat_size = 4
# Will use this when testing such that it can be looped efficiently
#mat_size = int( sys.argv[1])
#iteration = int( sys.argv[2])

mat_A = np.random.rand(mat_size,mat_size)
mat_B = np.random.rand(mat_size,mat_size)
print(mat_A)

np.save('mat_A', mat_A)
np.save('mat_B', mat_B)

A = np.load('mat_A.npy')
B = np.load('mat_B.npy')
print(mat_A)
