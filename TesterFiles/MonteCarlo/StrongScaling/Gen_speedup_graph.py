from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()

speedup_df = pd.read_pickle("speedup_df_strong_scaling.pkl")
speedup_df = speedup_df.T

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Runtime Speedup")
plt.legend()
plt.savefig(f"{p.parent.parent.parent}\Figures\MonteCarlo\montecarlo_speedup_strong_scaling.png")
