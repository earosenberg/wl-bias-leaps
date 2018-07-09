import numpy as np
import galsim
from scipy.optimize import curve_fit
import time
from galsim.gsparams import GSParams
from galsim.hsm import HSMParams
from scipy import fftpack
from scipy import interpolate

def mag_hist_power_law(x,norm):
      return norm* 10**(-8.85177133 +  0.716305260*x -0.00832345561*x*x)

## fnc is any callable functions defined
def draw_samples_specified_distribution(nsamples, fnc, z,*args):
    #zmin=1e-5
    #zmax=20
    #z = numpy.linspace(zmin,zmax) # Example indep variable
    fofz = fnc(z,*args)
    cdf_fofz = fofz.cumsum() # Next few steps to calculate CDF
    cdf_fofz -= cdf_fofz[0] # w/appropriate normalization
    cdf_fofz /= cdf_fofz[-1]
    max_val = np.where(cdf_fofz==1)[0].min()+1 #Nothing in CDF above 1
    cdf_fofz = cdf_fofz[:max_val] # cut off the trailing ones
    z = z[:max_val]
    model = interpolate.InterpolatedUnivariateSpline(cdf_fofz,z,k=4)
    np.random.seed(seed=10)
    samples = model(np.random.random(nsamples)) # Draw samples
    return samples

def magnitude(F,m0):
    #Convert from flux to magnitude
    return -2.5*np.log10(F)+m0
def flux(m,m0):
    #Convert from magnitude to flux
    return 10**((m-m0)/(-2.5))

def measureShape(noise_im, image_psf, disk_re, psf_sigma, pixel_scale):
        results = galsim.hsm.EstimateShear(noise_im,image_psf,strict=False,\
                                           guess_sig_gal=disk_re/pixel_scale, guess_sig_PSF=psf_sigma/pixel_scale,\
                                           shear_est='REGAUSS')
        if results.correction_status==0:
            return [results.corrected_e1, results.corrected_e2, results.corrected_shape_err,0]
        #if shape estimation unsuccessful
        else:
            if 'NaN' in results.error_message:
                em = -11
            elif 'adaptive' in results.error_message:
                em = -9
            elif 'min/max' in results.error_message:
                em = -7
            else:
                print results.error_message
                em = -999

            return [np.nan, np.nan, np.nan,em]
    
def addGaussianNoiseSNR(im,clean_im,snr):
    noise = galsim.GaussianNoise(sigma=1)
    im.addNoiseSNR(noise, snr, preserve_flux=True)
    noise_im = im - clean_im
    return noise_im

def addGaussianNoise(im,clean_im,sigma):
    noise = galsim.GaussianNoise(sigma=sigma)
    im.addNoise(noise)
    noise_im = im - clean_im
    return noise_im

def matchRotatedNans(obs):
    '''If any of the rotations of a single galaxy are nan, make all rotations nan'''
    tempObs = obs.transpose(0,1,3,2,4) #(SNR, ngal, shear, nrot, (e1, e2))
    for i in range(tempObs.shape[0]):
        for j in range(tempObs.shape[1]):
            for k in range(tempObs.shape[2]):
                if np.any(np.isnan(tempObs[i,j,k])):
                    for m in range(tempObs.shape[3]):
                        tempObs[i,j,k,m] = [np.nan] * tempObs[i,j,k,m].size
    obs = tempObs.transpose(0,1,3,2,4) 
    return obs

def intrinsic_shear(galaxy,shape_parameter_type,mag,ph):
    if shape_parameter_type == 'ellipticity':
        newGal = galaxy.shear(g=mag, beta=ph*galsim.radians)
    elif shape_parameter_type == 'distortion':
        newGal = galaxy.shear(e=mag, beta=ph*galsim.radians)
    elif shape_parameter_type == 'axis_ratio':
        newGal = galaxy.shear(q=mag, beta=ph*galsim.radians)
    else:
        raise ValueError, 'shape_parameter_type "%s" not recognized. Use one of the allowed keywords.' %shape_parameter_type
    return newGal


def drawGalaxies(shearedGals, noise_sigma, nopixel, sim_params, save_ims=False):
    '''
    Returns array of shape (ngal, nrot, #shears, (e1corr, e2corr))
    shearedGals is an array of shape (#shears, ngals*nrot) giving the sheared galaxy objects
    noise_sigma is a float giving the sigma of the Gaussian noise. If it is negative no noise is applied.
    nopixel is a bool. If True "no_pixel" is the method used to draw images, else "auto"
    sim_params is a list containing [psf, image_psf, psf_sigma, pixel_scale, gal_size, nrot, ngal]
    psf is the psf object
    psf_sigma is a number giving characteristic psf size, passed to shape measurement algorithm  
    pixel_scale is a float giving as/px
    gal_size is a number giving characteristic galaxy size, passed to shape measurement algorithm
    nrot is an int giving the number of rotations
    ngal is the number of galaxies
    save_ims is a bool indicating if images should be saved. Default False. Uses lots of memory if True!
    If True, also returns an array of galaxy images
    '''
    psf, image_psf, psf_sigma, pixel_scale, gal_size, nrot, ngal = sim_params
    if nopixel:
        method = 'no_pixel'
    else:
        method = 'auto'
    obs = []
    shearedGalT = shearedGals.transpose()
    #Iterate over rows of the same galaxy w/ different shears
    noisyGals=[]
    for galRowNum,galRow in enumerate(shearedGalT):
        gal_ims = []
        for i,gal in enumerate(galRow):
            #Convolve w/ PSF and draw image
            final = galsim.Convolve([gal, psf])
            if i==0:
                image = final.drawImage(scale=pixel_scale, method=method)
            else:
                image_shape = gal_ims[0].array.shape
                im_final = galsim.ImageF(*image_shape)
                image = final.drawImage(image=im_final, scale=pixel_scale, method=method)
            gal_ims.append(image)

        if noise_sigma >= 0: #negative noise_sigma taken to mean turn off noise
            #Add noise
            first = gal_ims[0].copy()
            noise_im = addGaussianNoise(first, gal_ims[0], noise_sigma)
            noisyImRow = [im + noise_im for im in gal_ims]
        else:
            noisyImRow = gal_ims
            
        if save_ims:
            noisyGals.append(noisyImRow)
        #Measure shape
        obsRow = [measureShape(noisyIm, image_psf, disk_re=gal_size[i], \
                                  psf_sigma=psf_sigma, pixel_scale=pixel_scale) for noisyIm in noisyImRow]

        obs.append(obsRow)

    obs = np.array(obs)
    obs=np.stack([obs[ngal*i:ngal*(i+1)] for i in range(nrot)]) #Put rotated versions in new axis
    #nrot, ngal, snr, shear, meas
    obs = obs.transpose(1,0,2,3) #Get desired shape
    error_message = obs[:,:,:,-1]
    shape_err = obs[:,:,:,-2]
    obs = obs[:,:,:,:-2] #remove shape error from obs
    #print obs.shape
    #Save file
    # outfile2 = open('../shear_bias_outputs/obs.pkl','wb')
    # cPickle.dump(obs,outfile2)
    # outfile2.close()
    if save_ims:
        return obs, noisyGals
    else:
        return obs
    #obs shape:
    #(ngal, nrot, #shears, 2)
    #Axis 1: Galaxy #
    #Axis 2: Rotated versions
    #Axis 3: Shear
    #Axis 4: shape measurement: [e1corr, e2corr] or [nan, nan] in case of failure