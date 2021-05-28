from multiprocessing import Pool
import pandas as pd
import random
import time
import sys

def monte_carlo(attempts):
    i = 0
    hits = 0
    for i in range(attempts):
        #Generates random points for x,y
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        #Checks whether (x,y) lies within unit circle
        if x**2 + y**2 <= 1:
            #if so, increments hits counter by 1
            hits += 1
    #Returns the number of his that we found
    return hits


def gen_time_results(attempts, no_cores, scaling_type):
    print(f"{no_cores} core(s)")
    #If we are running the "strong scaling" version, each core will perform N/P attempts
    if scaling_type == "strong":
        attempts_split = (10**attempts)//no_cores
    #If we are running the "weak scaling" version, each core will perform N attempts
    else:
        attempts_split = 10**attempts
    #Starts the timer
    start = time.perf_counter()
    #Creates a list to send the number of attempts to each core
    send_list = [(attempts_split,) for _ in range(no_cores)]
    #Opens a pool of worker processes
    p = Pool(processes=no_cores)
    #This is where the parallelism is taking place, each worker is given their number of attemps. The "monte_carlo" function will be called on each of them and they will return their number of "hits"
    hits_list = p.starmap(monte_carlo, (send_list))
    #Closes the pool of processes
    p.close()
    #Stops the timer
    finish = time.perf_counter()
    #Sums all of the hits from the workers
    total_hits = sum(hits_list)
    #Outputs the approximation of Pi to the user
    approx_pi = 4*total_hits/(attempts*no_cores)
    print(f"Estimation: {approx_pi}")
    #Calculates the time taken of the monte carlo calculation
    time_taken = round(finish-start,10)
    #Returns the time taken to be saved to a dataframe
    return time_taken
