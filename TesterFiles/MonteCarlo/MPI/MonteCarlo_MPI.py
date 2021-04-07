from mpi4py import MPI
import pandas as pd
import numpy as np
import random
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

attempts_power = int(sys.argv[1])
attempts = 10**attempts_power

def monte_carlo(attempts):
    i = 0
    hits = 0
    for i in range(0,attempts):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            hits += 1
    return hits

total_start = MPI.Wtime()

no_attempts = comm.bcast(attempts//size,root=0)

calc_start = MPI.Wtime()

hits = monte_carlo(no_attempts)

calc_finish = MPI.Wtime()

total_hits = comm.reduce(hits,op=MPI.SUM,root=0)
if rank == 0:
    approx_pi = 4*total_hits/attempts
    print(f"pi = {approx_pi}")

total_finish = MPI.Wtime()

proc0_total_start = comm.bcast(total_start,root=0)

time_difference = proc0_total_start - total_start

total_start += time_difference
calc_start += time_difference
calc_finish += time_difference
total_finish += time_difference

#print(f"rank: {rank}, total_start: {total_start}, calc_start: {calc_start}, calc_finish: {calc_finish}, total_finish: {total_finish}, time_difference: {time_difference}")

total_start_min = comm.reduce(total_start, op=MPI.MIN, root=0)
calc_start_min = comm.reduce(calc_start, op=MPI.MIN, root=0)
calc_finish_max = comm.reduce(calc_finish, op=MPI.MAX, root=0)
total_finish_max = comm.reduce(total_finish, op=MPI.MAX, root=0)

if rank == 0:
    scatter_time = calc_start_min - total_start_min
    calc_time = calc_finish_max - calc_start_min
    gather_time = total_finish_max - calc_finish_max
    total_time = total_finish_max - total_start_min
    assert np.isclose(scatter_time+calc_time+gather_time,total_time)
    #Must update this with whatever the max is in the bash file
    scatter_df = pd.read_pickle("scatter_df.pkl")
    calc_df = pd.read_pickle("calc_df.pkl")
    gather_df = pd.read_pickle("gather_df.pkl")
    total_df = pd.read_pickle("total_df.pkl")
    max_cores = 32
    core_list = [2**j for j in range(int(np.log2(max_cores))+1)]
    if size == 1:
        #add a new line with a new val at the left
        scatter_df = scatter_df.append( pd.DataFrame([[scatter_time if i==0 else 0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[attempts_power]) )
        calc_df = calc_df.append( pd.DataFrame([[calc_time if i==0 else 0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[attempts_power]) )
        gather_df = gather_df.append( pd.DataFrame([[gather_time if i==0 else 0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[attempts_power]) )
        total_df = total_df.append( pd.DataFrame([[total_time if i==0 else 0 for i in range(int(np.log2(max_cores))+1)]],columns=core_list, index=[attempts_power]) )
    elif size > 1:
        #add new value at right place
        size_power = int(np.log2(size))
        scatter_df.iloc[-1, size_power] = scatter_time
        calc_df.iloc[-1, size_power] = calc_time
        gather_df.iloc[-1, size_power] = gather_time
        total_df.iloc[-1, size_power] = total_time
    print(f"scatter: {scatter_time}")
    print(f"calc: {calc_time}")
    print(f"gather: {gather_time}")
    print(f"total: {total_time}")
    scatter_df.to_pickle("scatter_df.pkl")
    calc_df.to_pickle("calc_df.pkl")
    gather_df.to_pickle("gather_df.pkl")
    total_df.to_pickle("total_df.pkl")
