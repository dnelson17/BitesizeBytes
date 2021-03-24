from matplotlib import pyplot as plt
import pandas as pd

speedup_df = pd.read_pickle("speedup_df_strong_scaling.pkl")
speedup_df = speedup_df.T

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Runtime Speedup")
plt.legend()
plt.savefig("montecarlo_speedup_strong_scaling.png", dpi=1000)
