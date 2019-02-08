"""
Load the data from the simulations where we kept h=3, SN=10 and varied the X/Y centre across the IFU
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas_tools as P

N_sims=1000
SNs=np.logspace(0, 1.5, N_sims)
labels, vals=np.genfromtxt('/Users/vaughan/Science/KCLASH/Halpha_Sizes/Halpha_sims/sims/outputs/SN/0000_Halpha_single_fit_params_changing_peak_SN.dat', unpack=True, dtype=str)

N_cols=len(labels[[0, 1, 3, 4, 5, 6]])


values=np.empty((N_cols, N_sims))



for i in tqdm(range(N_sims)):


    _, tmp=np.genfromtxt('/Users/vaughan/Science/KCLASH/Halpha_Sizes/Halpha_sims/sims/outputs/SN/SN_{0:03d}_Halpha_single_fit_params_changing_peak_SN.dat'.format(i), unpack=True)  
    vals=tmp[[0, 1, 3, 4, 5, 6]]

    values[:, i]=vals


#inputs
Xcen=7 #pixels
Ycen=7 #pixels
PA=45 #Degrees
ell=0.2 
h=3 #Pixels




results=pd.DataFrame(columns=labels[[0, 1, 3, 4, 5, 6]], data=values.T)


plt.style.use('publication')
fig, ax=plt.subplots(figsize=(14, 7))
ax.plot(SNs, results['ell']/ell, c='orange', label=r'$\epsilon$')
ax.plot(SNs, results['h']/h, c='dodgerblue', label='$R_{\mathrm{{disk}}}$')
ax.plot(SNs, results['X0']/Xcen, c='gold', label='X$_0$')

ax.axhline(1.0, c='k', linestyle='dashed')
ax.axvline(2.0, c='k', linestyle='dotted')

ax.set_xlabel('Input S/N')
ax.set_ylabel('Measured/Input')

df=P.load_FITS_table_in_pandas('/Data/KCLASH/KCLASH_data_V4.6_RemeasuredSizes.fits')
arr=df.S2N_mean_image_fitting.loc[df.field_sample_biased_mass|df.cluster_sample|df.outskirts]
n, bins, rects=ax.hist(arr, weights=[1./50]*len(arr), density=False, histtype='stepfilled', edgecolor='k', facecolor='grey', alpha=0.3, zorder=10, linewidth=3.0, label=None)
n, bins, rects=ax.hist(arr, weights=[1./50]*len(arr), density=False, histtype='stepfilled', edgecolor='k', facecolor='None', zorder=10, linewidth=3.0, label=None)


ax.set_xlim([-0.1, 20.0])
ax.legend(fontsize=20)
fig.savefig('SN.pdf', bbox_inches='tight')