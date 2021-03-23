from matplotlib import pyplot as plt
import pandas as pd

speedup_df = pd.read_pickle(f"speedup_df_strong_scaling.pkl")
speedup_df = speedup_df.T

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Runtime Speedup")
plt.legend()
plt.savefig(f"montecarlo_speedup_{program_type}_scaling.jpeg", dpi=1000)
