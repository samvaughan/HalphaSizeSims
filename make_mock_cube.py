import numpy as np 
import matplotlib.pyplot as plt 
import subprocess
from astropy.io import fits
from  ppxf import ppxf_util as util
import scipy.interpolate as si
import scipy.constants as const
import os

from KMOS_tools import cube_tools as C

def gaussian(x, x0, sigma, A):
    return A/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*(x-x0)**2/sigma**2)


def make_gaussian_spectrum(lamdas, continuum, z, sigma_kms, peak_flux):


    scale=sigma_kms/const.c*1000.0*np.sqrt(2*np.pi)
    specNew, logLam, velscale=util.log_rebin(np.array([lamdas[0], lamdas[-1]]), continuum)
    emission_line=gaussian(np.exp(logLam), 0.65626*(1+z), sigma_kms/const.c*1000.0, scale)

    #Hack to get the height roughly right...
    #Come back to this!
    ha_pixel=np.argmin(np.abs(lamdas-0.65626*(1+z)))
    continuum_at_emission_line=np.mean(continuum[ha_pixel-10:ha_pixel+10])
    log_spec=specNew+emission_line/continuum_at_emission_line*peak_flux

    interpolator=si.interp1d(np.exp(logLam), log_spec, bounds_error=False)
    lin_spec=interpolator(lamdas)
    lin_spec[~np.isfinite(lin_spec)]=np.nanmedian(lin_spec)
    #Make the spectrum by combining the continuum and emission line
    spectrum=continuum + lin_spec

    return spectrum


def write_config(X, Y, Pa, ell, I0, h, filename):
    with open(filename, 'w') as f:
        f.write('X0\t{}\t0,14\n'.format(X))
        f.write('Y0\t{}\t0,14\n'.format(Y))

        f.write('FUNCTION Exponential\n')
        f.write('PA\t{}\t0,14\n'.format(Pa))
        f.write('ell\t{}\t0,14\n'.format(ell))
        f.write('I_0\t{}\t0,14\n'.format(I0))
        f.write('h\t{}\t0,14\n'.format(h))





def make_mock_cube(Xcen, Ycen, PA, ell, I0, h, z, peak_flux, SN, sky_background_level, Ha_velocity_dispersion, outfolder='mock_galaxy', suffix=""):

    #Load a cube to get the right wavelength array
    hdu=fits.open(os.path.expanduser('~/z/Data/KCLASH/Data/Sci/Final/COMBINE_SCI_RECONSTRUCTED_41309.fits'))
    pri_header=hdu[0].header
    header=hdu[1].header
    noise_header=hdu[2].header
    N_oversample=1

    N_lamdas, N_y, N_x=hdu[1].data.shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Make the model galaxy image

    #Write an imfit config file
    write_config(Xcen, Ycen, PA, ell, I0, h, '{}/config.txt'.format(outfolder))

    nrows=N_x*N_oversample
    ncols=N_y*N_oversample

    model_image_name='modelimage.fits'
    imfit_command = 'makeimage {0}/config.txt --nrows={1} --ncols={2} --psf={0}/Halpha_PSF.fits --output={0}/{3}'.format(outfolder, nrows, ncols, model_image_name)
    proc = subprocess.check_output(imfit_command.split(), stderr=subprocess.STDOUT)
    hdu=fits.open("{}/{}".format(outfolder, model_image_name))
    img=hdu[0].data
    img/=np.max(img)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Sky background- no skylines here
    #Come back to this?
    sky_background=np.abs(np.random.rand(N_lamdas, N_x*N_oversample, N_y*N_oversample)*sky_background_level)

    #Add noise
    #noise=np.random.randn(N_lamdas, N_x*N_oversample, N_y*N_oversample)/SN_image+sky_background
    #noisy_img=img+noise

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Make the spectrum, with velocity model?

    # #Velocity model
    # PA=0
    # ell=0.2
    # x_centre=7
    # y_centre=7
    # scale=3.5 #This is arctan scale lengths, ie do we get to the turnover?
    # Vmax=100.0

    # PA_rad=PA*np.pi/180.0
    # #-2 and 2 here are to get to the arctan turnover. Could improve!
    # x=(np.arange(ncols)-x_centre)/scale
    # y=(np.arange(nrows)-y_centre)/scale
    # X, Y=np.meshgrid(x, y)

    # X_r=np.cos(PA_rad)*X - np.sin(PA_rad)*Y
    # Y_r=np.sin(PA_rad)*X + np.cos(PA_rad)*Y
    # R=np.sqrt(X_r**2+ell*Y_r**2)

    # V=np.arctan(R)*Vmax



    lamdas=header['CRVAL3']+(np.arange(N_lamdas)+1)*header['CDELT3']


    #Continuum- do this better. Maybe take one from the cube fitting?
    polynomial=(-lamdas+0.4)*(lamdas-1.2)
    continuum=polynomial/np.max(polynomial)

    #H-alpha Gaussian 
    #Convert to log wavelengths, keep a fixed \Ha width, then convert back
    spectrum=make_gaussian_spectrum(lamdas, continuum, z, Ha_velocity_dispersion, peak_flux)


    #noisy_continuum=continuum +np.random.randn(N_lamdas)*np.max(continuum)/SN_image



    #Make the 3D cube
    cube=spectrum[:, None, None]*img[None, :, :]
    noise_cube=cube/SN#(cube+sky_background_level)
    nc=np.random.randn(*cube.shape)*noise_cube

    final_cube=(cube+nc+sky_background)*1e-19
    final_noise_cube=noise_cube*1e-19
    #noise_cube=((noisy_continuum)[:, None, None]*noisy_img[None, :, :])*1e-18/SN_image




    # #Downsample to KMOS resolution. Maybe do this at the beginning?
    # KMOS_res_cube=np.sum(np.sum(cube.reshape(N_lamdas, N_x, N_oversample, N_y, N_oversample), axis=4), axis=2)
    # KMOS_res_noise_cube=KMOS_res_cube/SN
    # nc=np.random.randn(*KMOS_res_noise_cube.shape)*KMOS_res_noise_cube



    #Write to a file

    header['EXTNAME']='SimCube.DATA'
    noise_header['EXTNAME']='SimCube.NOISE'
    header['CTYPE3'] = 'WAVE'
    header['specz'] = z

    data_HDU=fits.ImageHDU(final_cube, header=header)
    noise_HDU=fits.ImageHDU(final_noise_cube, header=noise_header)
    hdulist=fits.HDUList([fits.PrimaryHDU(header=pri_header), data_HDU, noise_HDU])
    hdulist.writeto('{}/SimulatedCube_{}.fits'.format(outfolder, suffix), overwrite=True)


    KMOS_cube_object=C.Cube('{}/SimulatedCube_{}.fits'.format(outfolder, suffix), is_sim=True)

    return KMOS_cube_object




if __name__=='__main__':

    from KMOS_tools import cube_tools as C

    # # #Settings
    # # #Units of the noise! So 
    # # N_oversample=1
    # # N_x=14
    # # N_y=14
    # # N_lamdas=2048
    # Xcen=7 #pixels
    # Ycen=5 #pixels
    # PA=45 #Degrees
    # ell=0.2 
    # I0=1 
    # h=3 #Pixels

    # z=0.4 #Redshift
    # peak_SN=5 #times the noise
    # # SN_image=5
    # # SN_continuum=5
    # # SN_emission_line=5
    # sky_background_level=0.01 #times the continuum level
    # Ha_velocity_dispersion=100 #km/s

    #c, n=make_mock_cube(Xcen, Ycen, PA, ell, I0, h, z, peak_SN, sky_background_level, Ha_velocity_dispersion, outfolder='mock_galaxy')

    #cube=C.Cube('mock_galaxy/SimulatedCube.fits', is_sim=True)
