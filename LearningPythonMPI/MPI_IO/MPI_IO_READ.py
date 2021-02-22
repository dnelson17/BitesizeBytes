from mpi4py import MPI
import numpy as np
import sys

# /usr/bin/mpiexec -n 1 python3 MPI_IO_READ.py 8

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mat_size = int(sys.argv[1])

amode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

buffer = np.empty((mat_size,mat_size), dtype='i')
displacement = 0
fh.Set_view(displacement)
fh.Read_all(buffer)
fh.Close()

print(buffer)
#print(np.transpose(buffer))
