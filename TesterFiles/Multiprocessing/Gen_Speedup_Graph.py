from matplotlib import pyplot as plt
import pandas as pd

program_type = "lapack"
#program_type = "myfunc"

speedup_df = pd.read_pickle(f"speedup_df_{program_type}.pkl")
print(speedup_df)

speedup_df = speedup_df.drop([32,64])
print(speedup_df)

speedup_df = speedup_df.T
print(speedup_df)

speedup_df.plot()
plt.xlabel("Number of Cores")
plt.ylabel("Runtime Speedup")
plt.legend()
#plt.show()
plt.savefig(f"multiprocessing_speedup_{program_type}.jpeg", dpi=1000)
