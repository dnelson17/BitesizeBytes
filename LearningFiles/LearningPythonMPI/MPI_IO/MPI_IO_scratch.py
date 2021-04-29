from mpi4py import MPI
import numpy as np

# /usr/bin/mpiexec -n 4 python3 MPI_IO_scratch.py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode = MPI.MODE_RDONLY
#fh = MPI.File.Open(comm, "mat_A/mat_A_8_1.txt", amode)
fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

item_count = 4

buffer = np.empty(item_count, dtype='i')

print(f"comm.Get_rank(): {comm.Get_rank()}, buffer.nbytes: {buffer.nbytes}, {comm.Get_rank()*buffer.nbytes}")

#filetype = MPI.INT.Create_vector(item_count, 1, size)
#filetype.Commit()

displacement = MPI.INT.Get_size()*item_count*rank
#fh.Set_view(displacement, filetype=filetype)
fh.Set_view(displacement)

fh.Read_all(buffer)

print(f"rank: {rank}, displacement: {displacement}, buffer: {buffer}")

#filetype.Free()
fh.Close()


buf_list = comm.gather(buffer,root=0)

if rank == 0:
    fin = np.vstack(buf_list)
    print(fin)
    print(np.transpose(fin))
