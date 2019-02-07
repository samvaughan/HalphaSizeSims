import numpy as np
import pandas as pd
from tqdm import tqdm

labels, vals=np.genfromtxt('sims/outputs/0001_Halpha_single_fit_params_changing_peak_SN.dat', unpack=True, dtype=str)

N_cols=len(labels[[0, 1, 3, 4, 5, 6]])
N_sims=1000



values=np.empty((N_cols, N_sims))

for i in tqdm(range(N_sims)):
    _, tmp=np.genfromtxt('sims/outputs/{0:04d}_Halpha_single_fit_params_changing_peak_SN.dat'.format(i), unpack=True)  
    vals=tmp[[0, 1, 3, 4, 5, 6]]

    values[:, i]=vals

results=pd.DataFrame(columns=labels[[0, 1, 3, 4, 5, 6]], data=values.T)