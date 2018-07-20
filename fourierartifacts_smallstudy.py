import time
import numpy as np

import galsim
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
#from galsim_test_helpers import *


# In[2]:


gsparams = galsim.GSParams(minimum_fft_size=8192,maximum_fft_size=8192*4)#,kvalue_accuracy=1.e-7,xvalue_accuracy=1.e-7)#,folding_threshold=5.e-3,stepk_minimum_hlr=5)

pixel_scale = 0.1
lamda = 550 #nm
diameter = 1.2 #m
fov = 70 #as
oversample = 10.
airy = galsim.Airy(lam=lamda, diam=diameter, scale_unit=galsim.arcsec, obscuration=0.3,gsparams=gsparams)
pixel = galsim.Pixel(pixel_scale,gsparams=gsparams)
psf = galsim.Convolve(airy, pixel,real_space=False)

#psfii_size = 2048
psfii_size = fov / (pixel_scale/oversample)
print psfii_size
image_psf=psf.drawImage(scale=pixel_scale,method='no_pixel')

oversampled_image_psf=galsim.ImageF(psfii_size,psfii_size)
oversampled_image_psf=psf.drawImage(image=oversampled_image_psf, scale=pixel_scale/10.,method='no_pixel')
psfii = galsim.InterpolatedImage(oversampled_image_psf,gsparams=gsparams)
psfiiIm = psfii.drawImage(image=oversampled_image_psf.copy(),scale=pixel_scale/10.,method='no_pixel')


# plt.imshow(oversampled_image_psf.array,norm=LogNorm())
# plt.colorbar();plt.show()
# plt.imshow(psfiiIm.array,norm=LogNorm())
# plt.colorbar()
# np.max(np.abs(psfiiIm.array - oversampled_image_psf.array))


gal = galsim.Gaussian(half_light_radius = 2, flux=100,gsparams=gsparams)

dil=2

obs = galsim.Convolve(gal, psf)
obsIm = obs.drawImage(scale=pixel_scale, method='no_pixel')
print(obsIm.array.shape)
obsIm = galsim.ImageF(obsIm.array.shape[0]*dil, obsIm.array.shape[1]*dil)
obsIm = obs.drawImage(image=obsIm, scale=pixel_scale, method='no_pixel')

vmin=np.min(obsIm.array)
vmax=np.max(obsIm.array)

obs = galsim.Convolve(gal, psfii,real_space=False)
obsImii = obs.drawImage(scale=pixel_scale, method='no_pixel')
print(obsImii.array.shape)
obsImii = galsim.ImageF(obsIm.array.shape[0], obsIm.array.shape[1])
obsImii = obs.drawImage(image=obsImii, scale=pixel_scale, method='no_pixel')


# fig,ax=plt.subplots(1,2)
# f1=ax[0].imshow(obsIm.array,norm=LogNorm(),vmin=vmin,vmax=vmax)
# f2=ax[1].imshow(np.abs(obsImii.array),norm=LogNorm(),vmin=vmin,vmax=vmax)
# fig.colorbar(f1,ax=ax[0])
# fig.colorbar(f2,ax=ax[1])


plt.imshow(obsImii.array/obsIm.array-1.,cmap=cm.bwr)
#plt.axhline(obsImii.array.shape[0]/4)
#plt.axhline(1024/10/2)
plt.colorbar()
plt.show()



