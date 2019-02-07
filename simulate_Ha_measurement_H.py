import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os

#Add path of our modules so we can import them
import sys
sys.path.append('/Users/vaughan/Science/KCLASH/Halpha_Sizes')
import measure_Halpha_size as MH
import make_mock_cube as MG

import argparse
parser = argparse.ArgumentParser(description='Simulate H-alpha measurements with varying size at a given S/N')
parser.add_argument('SN', type=float)

args = parser.parse_args()
SN=args.SN

N_samples=1000
Hs=np.linspace(0.3, 20, N_samples)


if not os.path.exists('sims/outputs/h/SN_{}'.format(int(SN))):
    os.makedirs('sims/outputs/h/SN_{}'.format(int(SN)))

for i, h in enumerate(tqdm(Hs)):

    
    Xcen=7 #pixels
    Ycen=7 #pixels
    PA=45 #Degrees
    ell=0.2 
    I0=1 
    

    z=0.4 #Redshift
    peak_flux=2 #Times the continuum level
    SN=SN #times the noise
    sky_background_level=0.1 #Units of the peak continuum level
    Ha_velocity_dispersion=100.0#km/s



    cube=MG.make_mock_cube(Xcen, Ycen, PA, ell, I0, h, z, peak_flux, SN, sky_background_level, Ha_velocity_dispersion, outfolder='/Users/vaughan/Science/KCLASH/Halpha_Sizes/Halpha_sims/mock_galaxy', suffix="H_SN_{}".format(int(SN)))

    # max_val=np.nanmax(cube.data)
    # cube.data=cube.data/max_val*1e-19
    # cube.noise=cube.noise/max_val*1e-19

    galaxy_directory='sims'
    #Dictionary of all the filenames
    filename_dict={'Halpha_image_filename':'{0}/Halpha_image_H_SN_{1}.fits'.format(galaxy_directory, SN), 
    'Halpha_config':'Halpha_config.txt'.format(galaxy_directory),
    'Halpha_mask_filename':'{0}/Halpha_mask_H_SN_{1}.fits'.format(galaxy_directory, SN),
    'Halpha_noise_filename':'{0}/Halpha_noise_H_SN_{1}.fits'.format(galaxy_directory, SN),
    'Halpha_psf_filename':'/Users/vaughan/Science/KCLASH/Halpha_Sizes/imfit/MACS1931_BCG_59407/Halpha_PSF.fits',#Just picked one
    'Halpha_model_output_filename':'{0}/Halpha_model_H_SN_{1}.fits'.format(galaxy_directory, SN),
    'Halpha_residual_output_filename':'{0}/Halpha_residual_H_SN_{1}.fits'.format(galaxy_directory, SN),
    'Halpha_params_output_filename':'{0}/outputs/h/SN_{2}/H_{1:03d}_Halpha_single_fit_params_changing_H_SN_{2}.dat'.format(galaxy_directory, i, int(SN))
    }

    #Get a mask around H-alpha
    spec_mask=cube.get_spec_mask_around_wave(cube.Ha_lam, width=0.002)
    
    MH.prepare_input_fits_files_Halpha(cube, filename_dict, spec_mask, make_mask=True)


    imfit_command_Halpha=MH.make_imfit_command(image=filename_dict['Halpha_image_filename'], config=filename_dict['Halpha_config'], mask=filename_dict['Halpha_mask_filename'], noise=filename_dict['Halpha_noise_filename'],
                 psf=filename_dict['Halpha_psf_filename'], model_output=filename_dict['Halpha_model_output_filename'], residual_output=filename_dict['Halpha_residual_output_filename'], params_output=filename_dict['Halpha_params_output_filename'], 
                 N_bootstraps=None, bootstrap_output=None, sky=0, silent=True)

    ret_code_Halpha=MH.run_imfit(imfit_command_Halpha)