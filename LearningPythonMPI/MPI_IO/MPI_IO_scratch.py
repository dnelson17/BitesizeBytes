from mpi4py import MPI
import numpy as np

# /usr/bin/mpiexec -n 4 python3 MPI_IO_scratch.py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode = MPI.MODE_RDONLY
#fh = MPI.File.Open(comm, "mat_A/mat_A_8_1.txt", amode)
fh = MPI.File.Open(comm, "./datafile.contig", amode)

item_count = 10

buffer = np.empty(item_count, dtype=np.int)
#buffer[:] = rank

print(f"comm.Get_rank(): {comm.Get_rank()}, buffer.nbytes: {buffer.nbytes}, {comm.Get_rank()*buffer.nbytes}")

offset = comm.Get_rank()*buffer.nbytes
fh.Read_at_all(offset, buffer)

print(buffer)

fh.Close()
