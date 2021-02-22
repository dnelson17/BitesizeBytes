from mpi4py import MPI
import numpy as np
import sys

# /usr/bin/mpiexec -n 8 python3 MPI_IO_WRITE.py 8

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mat_size = int(sys.argv[1])

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

#Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
power = np.log2(size)/2
#represents the number of partitons that must be calculated in the result matrix C
i_len = int(2**(np.ceil(power)))
j_len = int(2**(np.floor(power)))
#Represents the size of each partiton in the i and j axis
i_size = int(mat_size/i_len)
j_size = int(mat_size/j_len)

factor = 2**(int(np.log2(size))%2)

i_coord = factor * rank // i_len
j_coord = rank % j_len

buffer = np.empty((i_size,j_size), dtype='i')
buffer[:] = float(rank)

filetype = MPI.FLOAT.Create_vector(j_size, i_size, mat_size)
filetype.Commit()

print(f"rank: {rank}, i_coord {i_coord}, j_coord: {j_coord}, i_size: {i_size}, j_size: {j_size}, disp: {mat_size*i_coord*i_size + j_coord*j_size}")

displacement = MPI.FLOAT.Get_size()*(mat_size*i_coord*i_size + j_coord*j_size)
fh.Set_view(displacement, filetype=filetype)

fh.Write_all(buffer)
filetype.Free()
fh.Close()


buf_list = comm.gather(buffer,root=0)

if rank == 0:
    print(np.vstack(buf_list))
