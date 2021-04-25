from matplotlib import pyplot as plt
import pandas as pd

norm_factor = 5
print(f"\n\nfactor: {norm_factor}")
norm_power = range(5,14)[norm_factor]
print(f"norm power: {norm_power}")

cubed_list = [((2**j)**3)/((2**norm_power)**3) for j in range(5,14)]
#sqaured_list = [((2**j)**2)/(5**2) for j in range(5,12)]
order_list = [2**j for j in range(5,14)]


time_df = pd.read_pickle(f"time_df_libraries.pkl")
time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()
print(time_df.to_string())
norm_df = (time_df.T).apply(lambda x: x/x.iloc[norm_factor], axis=1, result_type='expand')
norm_df = norm_df.drop(['My_function','My_function_(32_Cores)'])
cubed_df = pd.DataFrame([cubed_list],columns=order_list,index=["(n^3)/100"])
norm_df = norm_df.append( cubed_df )
print(norm_df.to_string())
norm_df = norm_df.T
norm_df.plot()
plt.xlabel("Matrix Order")
plt.ylabel("Normalised Runtime")
plt.legend()
plt.show()
