import pandas as pd

core_list = [2**i for i in range(6)]
order_list = [2**i for i in range(8,16)]
my_func_df = pd.DataFrame([15.177072,119.339984,942.783879,7541.888572,None,None,None,None],columns=["My Function (Python)"],index=order_list)
my_func_32_df = pd.DataFrame([1.543031,7.949613,45.031003,349.793851,None,None,None,None],columns=["My Function (32 Cores Python)"],index=order_list)
#print(my_func_df.T)
#print(my_func_32_df.T)
total_df = (my_func_df.join(my_func_32_df))
total_df.to_pickle("time_df_my_function_no_lock.pkl")
