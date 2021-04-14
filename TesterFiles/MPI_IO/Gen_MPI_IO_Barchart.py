from matplotlib import pyplot as plt
import pandas as pd

core_list = [2**i for i in range(6)]

total_df = pd.read_pickle(f"time_dfs/total_df.pkl")
total_df = total_df.sort_index()
total_df = total_df.groupby(total_df.index).mean()

print(total_df)

normalise = lambda x, y: x/y

read_df = pd.read_pickle(f"time_dfs/read_df.pkl")
read_df = read_df.sort_index()
read_df = read_df.groupby(read_df.index).mean()

#print(read_df)

norm_read_df = read_df.combine(total_df, normalise)
print(norm_read_df)
norm_read_df = norm_read_df

calc_df = pd.read_pickle(f"time_dfs/calc_df.pkl")
calc_df = calc_df.sort_index()
calc_df = calc_df.groupby(calc_df.index).mean()

norm_calc_df = calc_df.combine(total_df, normalise)
print(norm_calc_df)
norm_calc_df = norm_calc_df


write_df = pd.read_pickle(f"time_dfs/write_df.pkl")
write_df = write_df.sort_index()
write_df = write_df.groupby(write_df.index).mean()

norm_write_df = write_df.combine(total_df, normalise)
print(norm_write_df)
norm_write_df = norm_write_df

sub_df = pd.concat([read_df,calc_df,write_df],keys=["read","calc","write"],axis=1)
print(sub_df.to_string())
#sub_df.plot(kind="bar",stacked=True)
#calc_df.plot()
#plt.show()
