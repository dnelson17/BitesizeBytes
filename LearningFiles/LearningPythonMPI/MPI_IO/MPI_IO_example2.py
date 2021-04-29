from mpi4py import MPI
import numpy as np

# /usr/bin/mpiexec -n 4 python3 MPI_IO_example2.py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

item_count = 4

buffer = np.empty(item_count, dtype='i')
buffer[:] = range(item_count*rank,item_count*(rank+1))

filetype = MPI.INT.Create_vector(item_count, 1, size)
filetype.Commit()

displacement = MPI.INT.Get_size()*rank
fh.Set_view(displacement, filetype=filetype)

fh.Write_all(buffer)
filetype.Free()
fh.Close()


buf_list = comm.gather(buffer,root=0)

if rank == 0:
    print(np.vstack(buf_list))
