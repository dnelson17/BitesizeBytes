from matplotlib import pyplot as plt
import pandas as pd

cubed_list = [((2**j)**3)/(100*((2**5)**3)) for j in range(5,12)]
#sqaured_list = [((2**j)**2)/(5**2) for j in range(5,12)]
order_list = [2**j for j in range(5,12)]


time_df = pd.read_pickle(f"time_df_libraries.pkl")
time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()
print(time_df.to_string())
norm_df = (time_df.T).apply(lambda x: x/x.iloc[0], axis=1, result_type='expand')
print(norm_df.to_string())
print(norm_df.index)
norm_df = norm_df.drop(['My_function','My_function_(32_Cores)'])
#squared_df = pd.DataFrame([sqaured_list],columns=order_list,index=["x^2"])
#norm_df = norm_df.append( squared_df )
cubed_df = pd.DataFrame([cubed_list],columns=order_list,index=["(n^3)/100"])
norm_df = norm_df.append( cubed_df )
print(norm_df.to_string())
norm_df = norm_df.T
norm_df.plot()
plt.xlabel("Matrix Order")
plt.ylabel("Normalised Runtime")
plt.legend()
plt.show()
