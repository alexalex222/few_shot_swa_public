# %%
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
filename = 'D:\\Temp\\torch_dataset\\regression\\malaria_df.hdf5'

df = pd.read_hdf(filename)

# %%
filename = 'D:\\Temp\\torch_dataset\\regression\\airfoil_self_noise.dat'
data = open(filename, "rb")
lines = data.readlines()
all_num = []
for line in lines:
    temp_str = str(line, "utf-8").split('\t')
    list_num = []
    for num in temp_str:
        list_num.append(float(num))
    all_num.append(list_num)
all_num_np = np.asarray(all_num)

# %%
fig, ax = plt.subplots()
ax.plot(all_num_np[:, 4])
plt.show()


