from matplotlib import pyplot as plt
import pandas as pd

program_type = "strong"
#program_type = "weak"

speedup_df = pd.read_pickle(f"speedup_df_{program_type}_scaling.pkl")
print(speedup_df)

speedup_df = speedup_df.T
print(speedup_df)

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Runtime Speedup")
plt.legend()
#plt.show()
plt.savefig(f"montecarlo_speedup_{program_type}_scaling.jpeg", dpi=1000)
