from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()

speedup_df = pd.read_pickle("speedup_df_weak_scaling.pkl")
speedup_df = speedup_df.drop([1,2,3,4])
speedup_df = speedup_df.T

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Normalised Runtime")
plt.legend()
plt.savefig(f"{p.parent.parent.parent}\Figures\MonteCarlo\montecarlo_speedup_weak_scaling.png")
