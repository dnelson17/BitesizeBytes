from multiprocessing import Pool
import pandas as pd
import numpy as np
import time
import sys

#def monte_carlo(attempts):
#    dots = np.square( np.random.rand(attempts,2) )
#    hits = 0
#    for pair in dots:
#        if np.sum(pair) <= 1:
#            hits += 1
#    return hits

def monte_carlo(attempts):
    i = 0
    hits = 0
    while i < attempts:
        dots = np.random.rand(1,2)
        if dots[0][0]**2 + dots[0][1]**2 <= 1:
            hits += 1
        i += 1
    return hits


def gen_time_results(attempts, no_cores, strong_scaling):
    print(f"{no_cores} core(s)")
    start = time.perf_counter()
    if strong_scaling:
        send_list = [(attempts//no_cores,) for _ in range(no_cores)]
    else:
        send_list = [(attempts,) for _ in range(no_cores)]
    p = Pool(processes=no_cores)
    hits_list = p.starmap(monte_carlo, (send_list))
    p.close()
    finish = time.perf_counter()
    total_hits = np.sum(hits_list)
    if strong_scaling:
        approx_pi = 4*total_hits/attempts
    else:
        approx_pi = 4*total_hits/(attempts*no_cores)
    print(f"Estimation: {approx_pi}")
    time_taken = round(finish-start,10)
    print(f"Time: {time_taken}\n")
    return time_taken


def main():
    attempts_list = [10**n for n in range(1,9)]
    core_list = [2**i for i in range(6)]
    for scaling_type in [True, False]:
        time_df = pd.DataFrame(columns=core_list)
        for attempts in attempts_list:
            print(f"Attempts = 10^{int(np.log10(attempts))}")
            new_times = []
            for no_cores in core_list:
                time_taken = gen_time_results(attempts, no_cores, scaling_type)
                new_times.append(time_taken)
            time_df = time_df.append( pd.DataFrame([new_times],columns=core_list,index=[int(np.log10(attempts))]) )
            if scaling_type:
                time_df.to_pickle("time_df_strong_scaling.pkl")
            else:
                time_df.to_pickle("time_df_weak_scaling.pkl")
            print("")
        print(f"\nOriginal times:\n{time_df}")
        time_df = time_df.sort_index()
        time_df = time_df.groupby(time_df.index).mean()
        print(f"\nTimes after ordering and mean:\n{time_df}")
        if scaling_type:
            speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
            ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
        else:
            speedup_df = time_df.apply(lambda x: x/x.iloc[0], axis=1, result_type='expand')
            ideal_df = pd.DataFrame([[1 for _ in range(6)]],columns=core_list,index=["Ideal"])
        print(f"\nIdeal:\n{ideal_df}")
        speedup_df = speedup_df.append( ideal_df )
        print(f"\nSpeedups:\n{speedup_df}")
        if scaling_type:
            time_df.to_pickle("time_df_strong_scaling.pkl")
            speedup_df.to_pickle("speedup_df_strong_scaling.pkl")
        else:
            time_df.to_pickle("time_df_weak_scaling.pkl")
            speedup_df.to_pickle("speedup_df_weak_scaling.pkl")


if __name__ == '__main__':
    main()
