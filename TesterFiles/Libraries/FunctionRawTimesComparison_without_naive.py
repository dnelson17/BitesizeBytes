from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time
import sys

def main():
    lock = int(sys.argv[1])
    
    size_list = [2**i for i in range(10,16)]
    no_runs = 10
    if lock:
        time_df = pd.DataFrame(columns=["Numpy matmul (Lock)","Python dgemm (Lock)","Python sgemm (Lock)"])
    else:
        time_df = pd.DataFrame(columns=["Numpy matmul (No Lock)","Python dgemm (No Lock)","Python sgemm (No Lock)"])
    for mat_size in size_list:
        print(f"Mat size: {mat_size}")
        for i in range(no_runs):
            print(f"i: {i}")
            
            m1 = np.random.rand(mat_size,mat_size).astype(np.float32)
            m2 = np.random.rand(mat_size,mat_size).astype(np.float32)
            new_times=[]
            
            numpy_start = time.perf_counter()
            mn = np.matmul(m1,m2)
            numpy_finish = time.perf_counter()
            new_times.append(round(numpy_finish-numpy_start,8))
            
            dgemm_start = time.perf_counter()
            md = FB.dgemm(alpha=1.0, a=m1, b=m2)
            dgemm_finish = time.perf_counter()
            new_times.append(round(dgemm_finish-dgemm_start,8))
            
            sgemm_start = time.perf_counter()
            ms = FB.sgemm(alpha=1.0, a=m1, b=m2)
            sgemm_finish = time.perf_counter()
            new_times.append(round(sgemm_finish-sgemm_start,8))

            print(new_times)
            
            if lock:
                time_df.to_pickle("time_df_libraries_lock.pkl")
                time_df = time_df.append( pd.DataFrame([new_times],columns=["Numpy matmul (Lock)","Python dgemm (Lock)","Python sgemm (Lock)"],index=[mat_size]) )
            else:
                time_df.to_pickle("time_df_libraries_no_lock.pkl")
                time_df = time_df.append( pd.DataFrame([new_times],columns=["Numpy matmul (No Lock)","Python dgemm (No Lock)","Python sgemm (No Lock)"],index=[mat_size]) )
    print(f"\nOriginal times:\n{time_df}")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(f"\nTimes after ordering and mean:\n{time_df}")


if __name__ == '__main__':
    main()
