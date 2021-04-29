from multiprocessing import Pool
import pandas as pd
import random
import time
import sys

def monte_carlo(attempts):
    i = 0
    hits = 0
    for i in range(0,attempts):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            hits += 1
    return hits


def gen_time_results(attempts, no_cores, scaling_type):
    print(f"{no_cores} core(s)")
    if scaling_type == "strong":
        attempts_split = 10**attempts
    else:
        attempts_split = (10**attempts)//no_cores
    start = time.perf_counter()
    send_list = [(attempts_split,) for _ in range(no_cores)]
    p = Pool(processes=no_cores)
    hits_list = p.starmap(monte_carlo, (send_list))
    p.close()
    finish = time.perf_counter()
    total_hits = sum(hits_list)
    approx_pi = 4*total_hits/(attempts*no_cores)
    print(f"Estimation: {approx_pi}")
    time_taken = round(finish-start,10)
    print(f"Time: {time_taken}\n")
    return time_taken


def main():
    scaling_list = ["weak","strong"]
    for scaling_type in scaling_list:
        print(f"{scaling_type} scaling")
        attempts_power_list = [n for n in range(4,10)]
        core_list = [2**i for i in range(6)]
        time_df = pd.DataFrame(columns=core_list)
        for power in attempts_power_list:
            print(f"Attempts = 10^{power}")
            new_times = []
            for no_cores in core_list:
                time_taken = gen_time_results(attempts, no_cores, scaling_type)
                new_times.append(time_taken)
            time_df = time_df.append( pd.DataFrame([new_times],columns=core_list,index=[power]) )
            time_df.to_pickle(f"Time_dfs/time_df_{scaling_type}_scaling.pkl")


if __name__ == '__main__':
    main()
