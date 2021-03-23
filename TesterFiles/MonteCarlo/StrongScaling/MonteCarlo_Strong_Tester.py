from multiprocessing import Pool
import pandas as pd
import numpy as np
import time
import sys

def monte_carlo(attempts):
    i = 0
    hits = 0
    while i < attempts:
        dots = np.random.rand(1,2)
        if dots[0][0]**2 + dots[0][1]**2 <= 1:
            hits += 1
        i += 1
    return hits


def gen_time_results(attempts, no_cores):
    print(f"{no_cores} core(s)")
    start = time.perf_counter()
    send_list = [(attempts//no_cores,) for _ in range(no_cores)]
    p = Pool(processes=no_cores)
    hits_list = p.starmap(monte_carlo, (send_list))
    p.close()
    finish = time.perf_counter()
    total_hits = np.sum(hits_list)
    approx_pi = 4*total_hits/attempts
    print(f"Estimation: {approx_pi}")
    time_taken = round(finish-start,10)
    print(f"Time: {time_taken}\n")
    return time_taken


def main():
    attempts_list = [10**n for n in range(1,9)]
    core_list = [2**i for i in range(6)]
    time_df = pd.DataFrame(columns=core_list)
    for attempts in attempts_list:
        print(f"Attempts = 10^{int(np.log10(attempts))}")
        new_times = []
        for no_cores in core_list:
            time_taken = gen_time_results(attempts, no_cores)
            new_times.append(time_taken)
        time_df = time_df.append( pd.DataFrame([new_times],columns=core_list,index=[int(np.log10(attempts))]) )
        time_df.to_pickle("time_df_strong_scaling.pkl")
        print("")
    print(f"\nOriginal times:\n{time_df}")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(f"\nTimes after ordering and mean:\n{time_df}")
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    print(f"\nIdeal:\n{ideal_df}")
    speedup_df = speedup_df.append( ideal_df )
    print(f"\nSpeedups:\n{speedup_df}")
    time_df.to_pickle("time_df_strong_scaling.pkl")
    speedup_df.to_pickle("speedup_df_strong_scaling.pkl")


if __name__ == '__main__':
    main()
