"""
Load the data from the simulations where we kept h=3, SN=10 and varied the X/Y centre across the IFU
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


N_sims=1000
Hs=np.linspace(0.3, 20, N_sims)

R={}

for SN in [2, 5, 10]:

    labels, vals=np.genfromtxt('/Users/vaughan/Science/KCLASH/Halpha_Sizes/Halpha_sims/sims/outputs/h/SN_{0}/H_000_Halpha_single_fit_params_changing_H_SN_{0}.dat'.format(SN), unpack=True, dtype=str)

    N_cols=len(labels[[0, 1, 3, 4, 5, 6]])
    


    values=np.empty((N_cols, N_sims))



    for i in tqdm(range(N_sims)):


        _, tmp=np.genfromtxt('/Users/vaughan/Science/KCLASH/Halpha_Sizes/Halpha_sims/sims/outputs/h/SN_{0}/H_{1:03d}_Halpha_single_fit_params_changing_H_SN_{0}.dat'.format(SN, i), unpack=True)  
        vals=tmp[[0, 1, 3, 4, 5, 6]]

        values[:, i]=vals

    results=pd.DataFrame(columns=labels[[0, 1, 3, 4, 5, 6]], data=values.T)

    R['SN_{}'.format(SN)]=results


#Plotting
plt.style.use('publication')
fig, ax=plt.subplots(figsize=(9, 8))
ax.plot(Hs, R['SN_2'].loc[:, 'h']/Hs, label='S/N=2', c='darkgrey')
ax.plot(Hs, R['SN_5'].loc[:, 'h']/Hs, label='S/N=5', c='dodgerblue')
ax.plot(Hs, R['SN_10'].loc[:, 'h']/Hs, label='S/N=10', c='crimson')

ax.axhline(1.0, c='k', linestyle='dashed')
ax.axvline(7.0, c='k', linewidth=1.0) 

ax.legend(fontsize=20)
ax.set_xlabel('Input size (pixels)')
ax.set_ylabel('Measured size/Input size')

fig.savefig('H_tests.pdf', bbox_inches='tight')
plt.show()











