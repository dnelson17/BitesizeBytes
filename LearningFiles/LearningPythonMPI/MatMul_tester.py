from mpi4py import MPI
import time
import numpy as np
from scipy.linalg import blas as FB
import sys

# /usr/bin/mpiexec -n 5 python3 MatMul_tester.py 32 32

numberRows = int( sys.argv[1])
numberColumns = int( sys.argv[2])
TaskMaster = 0

assert numberRows == numberColumns

#print(numberRows)

#print ("Initialising variables.\n")
mat_A = np.random.rand(numberRows,numberColumns)
mat_B = np.random.rand(numberRows,numberColumns)
mat_C = np.zeros(shape=(numberRows, numberColumns))

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

#print(worldSize)

#print ("Process %d started.\n" % (rank))
#print ("Running from processor %s, rank %d out of %d processors.\n" % (processorName, rank, worldSize))

#Calculate the slice per worker
if (worldSize == 1):
    slice = numberRows
else:
    slice = int(numberRows / (worldSize-1)) #make sure it is divisible
assert slice >= 1


comm.Barrier()
    
if rank == TaskMaster:
    for i in range(1, worldSize):
        offset = (i-1)*slice #0, 10, 20
        row = mat_A[offset,:]
        comm.send(offset, dest=i, tag=i)
        comm.send(row, dest=i, tag=i)
        for j in range(0, slice):
            comm.send(mat_A[j+offset,:], dest=i, tag=j+offset)

comm.Barrier()

if rank != TaskMaster:
    offset = comm.recv(source=0, tag=rank)
    recv_data = comm.recv(source=0, tag=rank)
    for j in range(1, slice):
        mat_C = comm.recv(source=0, tag=j+offset)
        recv_data = np.vstack((recv_data, mat_C))
    #Loop through rows
    t_start = MPI.Wtime()
    for i in range(0, slice):
        res = np.zeros(shape=(numberColumns))
        if (slice == 1):
            r = recv_data
        else:
            r = recv_data[i,:]
        ai = 0
        for j in range(0, numberColumns):
            q = mat_B[:,j] #get the column we want
            for x in range(0, numberColumns):
                res[j] = res[j] + (r[x]*q[x])
            ai = ai + 1
        if (i > 0):
            send = np.vstack((send, res))
        else:
            send = res
    t_diff = MPI.Wtime() - t_start

    print(t_diff)
    #Send large data
    comm.Send([send, MPI.FLOAT], dest=0, tag=rank) #1, 12, 23


#comm.Barrier()

if rank == TaskMaster:
    res1 = np.zeros(shape=(slice, numberColumns))
    comm.Recv([res1, MPI.FLOAT], source=1, tag=1)
    kl = np.vstack((res1))
    for i in range(2, worldSize):
        resx= np.zeros(shape=(slice, numberColumns))
        comm.Recv([resx, MPI.FLOAT], source=i, tag=i)
        kl = np.vstack((kl, resx))

#comm.Barrier()
