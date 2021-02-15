from mpi4py import MPI
import time
import numpy as np
import sys

# ssh -XY 40199787@aigis.mp.qub.ac.uk
# 
# ssh aigis06
# cd BitesizeBytes/LearningPythonMPI/MPI_IO
# /usr/bin/mpiexec -n 4 python3 MatrixMult.py 4 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

#Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
power = np.log2(size)/2
#represents the number of partitons that must be calculated in the result matrix C
i_len = int(2**(np.ceil(power)))
j_len = int(2**(np.floor(power)))
i_size = int(mat_size/i_len)
j_size = int(mat_size/j_len)

# Initialize the 2 random matrices only if this is rank 0
if rank == 0:
    t_start = MPI.Wtime()
    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([int(i*i_size),int(j*j_size)])
else:
    send_list = None

info = comm.scatter(send_list,root=0)

f_A = MPI.File.Open(comm,"mat_A.txt", amode=MPI.MODE_RDONLY)
mat_A = np.empty((mat_size,info[0][0]-info[0][1]+1), dtype=np.byte)
f_B = MPI.File.Open(comm,"mat_B.txt", amode=MPI.MODE_RDONLY)
mat_B = np.empty((info[1][0]-info[1][1]+1,mat_size), dtype=np.byte)
mat_C = matrix_mult(mat_A,mat_B)

res_list = comm.gather(mat_C,root=0)

if rank == 0:
    res = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
    t_diff = MPI.Wtime() - t_start
    print(t_diff)
    #print(np.array_equal(res, ans))
    
