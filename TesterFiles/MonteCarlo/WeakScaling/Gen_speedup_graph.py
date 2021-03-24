from matplotlib import pyplot as plt
import pandas as pd

speedup_df = pd.read_pickle("speedup_df_weak_scaling.pkl")
speedup_df = speedup_df.drop([1,2,3,4])
speedup_df = speedup_df.T

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Normalised Runtime")
plt.legend()
plt.savefig("montecarlo_speedup_weak_scaling.png", dpi=1000)
