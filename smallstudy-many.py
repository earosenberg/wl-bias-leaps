import numpy as np
from matplotlib import pyplot as plt
import galsim
import time
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import shape_measurement as sm
import cPickle

def circularize(gal):
    inv=np.linalg.inv(gal.jac).flatten()
    newgal = gal.transform(*inv)
    return newgal

def circularizeAll(gal):
    if type(gal) != galsim.compound.Sum:
        return circularize(gal)
    else:
        bulge, disk = gal.obj_list
        return circularize(bulge) + circularize(disk)
 
def removeBulge(gal):
    if type(gal) != galsim.compound.Sum:
        return gal
    else:
        disk = gal.obj_list[1]
        return disk
    
gsparams = galsim.GSParams(kvalue_accuracy=1.e-5,maximum_fft_size=2048*10,maxk_threshold=9.e-4)

pixel_scale = 0.1 #as/px
lamda = 550 #nm
diameter = 1.2 #m
psf_oversample = 5.
gal_oversample = 2.
circ = True
bulgeHandling = 0 #0: Keep Bulge+Disk. 1: Replace with just disk. 2: Remove entire galaxy

airy = galsim.Airy(lam=lamda, diam=diameter, scale_unit=galsim.arcsec, obscuration=0.3, gsparams=gsparams)
pixel = galsim.Pixel(pixel_scale,gsparams=gsparams)
psf = galsim.Convolve(airy, pixel)

given_psf = psf.drawImage(scale=pixel_scale/psf_oversample,method='no_pixel') #Draw oversampled psf image
psfii = galsim.InterpolatedImage(given_psf, gsparams=gsparams)

#Load galaxy catalog and select galaxies
cc = galsim.COSMOSCatalog(dir='/disks/shear15/KiDS/ImSim/pipeline/data/COSMOS_25.2_training_sample/',use_real=False)
hlr, sn, q = [np.array([pc[4][i] for pc in cc.param_cat]) for i in range(1,4)]

small100I = np.where(np.logical_and(hlr*np.sqrt(q)>2.5, sn>=0.5))[0][:1000] #Large galaxies, reasonable sersic n
gals = cc.makeGalaxy(small100I, chromatic=False, gsparams=gsparams)


if bulgeHandling == 1:
    gals = [removeBulge(gal) for gal in gals] #Change bulge+disk to just disk

elif bulgeHandling == 2:
    gals = [gal for gal in gals if type(gal) != galsim.compound.Sum] #Remove bulge+disk profiles
        
if circ:
    gals = [circularizeAll(gal) for gal in gals]
    q = np.ones_like(sn)
ii=0
print len(gals)
sde1,sde2=[],[]
for gal in gals:
    ii+=1
    if ii%50==0: print ii
    fin = galsim.Convolve([gal,psf])
    given_im = fin.drawImage(scale=pixel_scale/gal_oversample, method='no_pixel')

    gal_interp = galsim.InterpolatedImage(given_im,gsparams=gsparams)
    inv_gauss = galsim.Deconvolve(psfii)
    dec = galsim.Convolve(gal_interp,inv_gauss)
    rec = galsim.Convolve(dec, psfii)

    recIm = rec.drawImage(scale=pixel_scale/gal_oversample,method='no_pixel')

    decIm = dec.drawImage(scale=pixel_scale/gal_oversample)
    ss=decIm.array.shape[0]
    stamp=galsim.ImageF(ss,ss)
    recsize = recIm.array.shape[0]
    origIm = galsim.ImageF(recsize,recsize)
    origIm = fin.drawImage(image=origIm,scale=pixel_scale/gal_oversample,method='no_pixel')

    orig_shapedata=galsim.hsm.FindAdaptiveMom(origIm,strict=False)
    rec_shapedata=galsim.hsm.FindAdaptiveMom(recIm,strict=False)
    if orig_shapedata.error_message:
        orig_shape_e1, orig_shape_e2 = np.nan, np.nan
    if rec_shapedata.error_message:
        rec_shape_e1, rec_shape_e2 = np.nan, np.nan
    else:
        orig_shape = orig_shapedata.observed_shape
        rec_shape = rec_shapedata.observed_shape
        orig_shape_e1, orig_shape_e2 = orig_shape.e1, orig_shape.e2
        rec_shape_e1, rec_shape_e2 = rec_shape.e1, rec_shape.e2

    sde1.append(np.abs(rec_shape_e1-orig_shape_e1))
    sde2.append(np.abs(rec_shape_e2-orig_shape_e2))

print 'Saving data'
res = [sde1,sde2, hlr[small100I], sn[small100I], q[small100I]]
basename = '/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/measErrs'
if circ:
    basename = basename + '_circ'
if bulgeHandling==1:
    basename = basename + '_diskonly'
if bulgeHandling==2:
    basename = basename + '_nobulgedisk'
name = basename + '.pkl'
fil=open(name,'wb')
cPickle.dump(res,fil)
fil.close()
