import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

randNum = numpy.zeros(1)

#This part is only executed by core 1
if rank == 1:
        #Generate a random number
        randNum = numpy.random.random_sample(1)
        print(f"Process {rank} drew the number: {randNum[0]}")
        #Send the number to core 0
        comm.Send(randNum, dest=0)
        #Receive the new number from core 0
        comm.Recv(randNum, source=0)
        print(f"Process {rank} received the number: {randNum[0]}")

##This part is only executed by core 0
if rank == 0:
        print(f"Process {rank} before receiving has the number: {randNum[0]}")
        #Receive the number from core 1
        comm.Recv(randNum, source=1)
        #Multiply the number by 2
        print(f"Process {rank} received the number: {randNum[0]}")
        randNum *= 2
        #Send the new number back to core 1
        comm.Send(randNum, dest=1)
