import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_pickle("MAG2P_nbonded-2025-2-6-16:25:37.pickle") 
pd.read_pickle("MAG2P_nbonded-2025-2-6-16:25:37.pickle") 
df = df.sort_values(["shift", "lambda"])   
df.groupby(["shift","lambda"])["mean_bonds"].mean().reset_index().values.astype(float) 

