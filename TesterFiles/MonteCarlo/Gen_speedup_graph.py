from matplotlib import pyplot as plt
import pandas as pd

#program_type = "strong"
program_type = "weak"

speedup_df = pd.read_pickle(f"speedup_df_{program_type}_scaling.pkl")
print(speedup_df)

if program_type == "weak":
    speedup_df = speedup_df.drop([1,2,3,4])

speedup_df = speedup_df.T
print(speedup_df)

speedup_df.plot()
plt.xlabel("Number of Cores")
if program_type == "strong":
    plt.ylabel("Runtime Speedup")
elif program_type == "weak":
    plt.ylabel("Runtime")
plt.legend()
#plt.show()
plt.savefig(f"montecarlo_speedup_{program_type}_scaling.jpeg", dpi=1000)
