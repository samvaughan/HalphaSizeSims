"""
Load the data from the simulations where we kept h=3, SN=10 and varied the X/Y centre across the IFU
"""


import numpy as np
import pandas as pd
from tqdm import tqdm



labels, vals=np.genfromtxt('sims/outputs/0001_Halpha_single_fit_params_changing_peak_SN.dat', unpack=True, dtype=str)
X, Y=np.indices((14, 14))


N_cols=len(labels[[0, 1, 3, 4, 5, 6]])
N_sims_per_xy=10
N_values=len(X.ravel())


values=np.empty((N_cols, N_values))



for j, (x, y) in tqdm(enumerate(zip(X.ravel(), Y.ravel())), total=len(X.ravel())):

    tmpvals=np.empty((N_cols, N_sims_per_xy))
    for i in range(N_sims_per_xy):
        _, tmp=np.genfromtxt('sims/outputs/XY/x_{0}_y_{1}_i_{2:03d}_Halpha_single_fit_params_changing_peak_SN.dat'.format(x, y, i), unpack=True)  
        vals=tmp[[0, 1, 3, 4, 5, 6]]

        tmpvals[:, i]=vals
    values[:, j]=np.mean(tmpvals, axis=1)

results=pd.DataFrame(columns=labels[[0, 1, 3, 4, 5, 6]], data=values.T)

