#Import libraries
import numpy as np
from matplotlib import pyplot as plt
import galsim
from scipy.optimize import curve_fit
import time
#from galsim.gsparams import GSParams
from galsim.hsm import HSMParams
from scipy import fftpack
import shape_measurement as sm

largehist = False
halfpixel = False
doubleStamp = True
oversamplePSF = True
obs_e = True
test=False
numDone=0
# for halfpixel in (False, True):
#     for doubleStamp in (False, True):
#         for oversamplePSF in (False, True):
numDone+=1
hp_name = 'halfpixel'
ds_name = 'doublestamp'
os_name = 'oversample'
obse_name = 'obs_e'
test_name='test'
flags = [halfpixel, doubleStamp, oversamplePSF, obs_e,test]
flagnames = [hp_name, ds_name, os_name, obse_name,test_name]

#Simulation options
shearList = [(0,0.01)] #list of shears to apply
#gsparams = galsim.GSParams(maximum_fft_size=12288)

#Parameters of the PSF
if halfpixel:
    pixel_scale = 0.1/2. #as/px
else:
    pixel_scale = 0.1

#Airy PSF
lamda = 550 #nm
diameter = 1.2 #m
airy = galsim.Airy(lam=lamda, diam=diameter, scale_unit=galsim.arcsec, obscuration=0.3)
pixel = galsim.Pixel(pixel_scale)
psf = galsim.Convolve(airy, pixel)
image_psf=psf.drawImage(scale=pixel_scale,method='no_pixel')
if oversamplePSF:
    oversampled_image_psf=psf.drawImage(scale=pixel_scale/10.,method='no_pixel')
else:
    oversampled_image_psf=image_psf


#Load galaxy catalog and select galaxies
cc = galsim.COSMOSCatalog(dir='/disks/shear15/KiDS/ImSim/pipeline/data/COSMOS_25.2_training_sample/',use_real=False)
hlr, sn, q = [np.array([pc[4][i] for pc in cc.param_cat]) for i in range(1,4)]

small100I = np.where(hlr*np.sqrt(q)>2.5)[0][:200] #Large galaxies
galaxies = cc.makeGalaxy(small100I, chromatic=False)#, gsparams=gsparams)


#Draw galaxy images
galIms=[]
for gal in galaxies:
    psfgal = galsim.Convolution([gal, psf])
    galIm = psfgal.drawImage(scale=pixel_scale, method='no_pixel')
    galIms.append(galIm)

if doubleStamp:
    stamp_scale = 2
else:
    stamp_scale = 1
base_images=[galsim.ImageF(gal.array.shape[0]*stamp_scale,gal.array.shape[1]*stamp_scale,scale=pixel_scale) for gal in galIms]
base_images2=[galsim.ImageF(gal.array.shape[0]*stamp_scale,gal.array.shape[1]*stamp_scale,scale=pixel_scale) for gal in galIms]


galIms=[]
for i,gal in enumerate(galaxies):
    psfgal = galsim.Convolution([gal, psf])
    galIm = psfgal.drawImage(image=base_images[i],scale=pixel_scale, method='no_pixel')
    galIms.append(galIm)


psfii = galsim.InterpolatedImage(oversampled_image_psf)


# In[8]:


shearedIm=[]
for i,gal in enumerate(galIms):
    gali = galsim.InterpolatedImage(gal) #Warning: these are large and take a lot of memory
    shRow = sm.cfGal(gali,psfii,shearList) # Warning: these are large and take a lot of memory
    shearedImRow = [sh.drawImage(image=base_images2[i],method='no_pixel',scale=pixel_scale) for sh in shRow]
    shearedIm.append(shearedImRow)
shearedIm=np.array(shearedIm)


# In[9]:


psf_sigma=1
if obs_e:
    obsOrig = np.array([sm.measureObsShape(gal,image_psf, hlr[i]*pixel_scale, psf_sigma, pixel_scale) for i,gal in enumerate(galIms)])
    obsConv = np.array([sm.measureObsShape(gal,image_psf, hlr[i]*pixel_scale, psf_sigma, pixel_scale) for i,gal in enumerate(shearedIm.transpose()[0])])
else:
    obsOrig = np.array([sm.measureShape(gal,image_psf, hlr[i]*pixel_scale, psf_sigma, pixel_scale) for i,gal in enumerate(galIms)])
    obsConv = np.array([sm.measureShape(gal,image_psf, hlr[i]*pixel_scale, psf_sigma, pixel_scale) for i,gal in enumerate(shearedIm.transpose()[0])])

e1,e2 = obsOrig.transpose()[0],obsOrig.transpose()[1]
e1c,e2c = obsConv.transpose()[0],obsConv.transpose()[1]


# In[10]:


ooT = obsOrig.transpose()
ocT = obsConv.transpose()
diffI = np.where(ooT[-1] != ocT[-1])[0]
sameI = np.where(np.logical_and(ooT[-1] == 0, ocT[-1] == 0.))[0]
for i in diffI:
    print ooT[-1,i], ocT[-1,i]
print 'number of failures original/conv: ', np.sum(ooT[-1] != 0),',', np.sum(ocT[-1] != 0)


# In[11]:


for i in diffI:
    print (ooT[-1,i] - ocT[-1,i])/9.
diffErr = np.array([(ooT[-1,i] - ocT[-1,i])/9. for i in diffI])
print diffErr
origErr = np.abs(np.sum(diffErr[diffErr<0]))
convError = np.abs(np.sum(diffErr[diffErr>0]))


# In[12]:


diffe1 = e1-e1c
diffe2 = e2-e2c
diffe1 = diffe1[sameI]
diffe2 = diffe2[sameI]
print min(diffe1),max(diffe1)
print min(diffe2),max(diffe2)


# In[13]:


fig,ax=plt.subplots(sharey=True,figsize=(8,6))

if largehist:#np.abs(min(diffe1))>0.1:
    b=ax.hist(diffe2,histtype='step',range=(-0.49,0.27),bins=38*2,linewidth=2,color='k',label='e2')
    a=ax.hist(diffe1,histtype='step',range=(-0.13,0.11),bins=12*2,linewidth=2,color='r',label='e1')
else:
    b=ax.hist(diffe2,histtype='step',range=(-0.06,0.05),bins=88,linewidth=2,color='k',label='e2')
    a=ax.hist(diffe1,histtype='step',range=(-0.06,0.05),bins=88,linewidth=2,color='r',label='e1')
ax.legend()

ax.set_xlabel('e_orig - e_conv',size=15)
plt.axvline(0,color='k',linestyle=':')
#ax.set_xlim(-0.4,0.3)
plt.tight_layout()
name = '/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/e_diff_histogram'
for i in range(len(flags)):
    if flags[i]:
        name += '-'+flagnames[i]
if largehist:
    name+='-largehist'
name+='.png'
plt.savefig(name)


# # In[15]:


shearedFlat = shearedIm.flatten()

from matplotlib.colors import LogNorm
x=50
zoom = False
vlim = False

if vlim:
    vmax = 0.1
else:
    vmax=None

norm = LogNorm()
fig,ax=plt.subplots(x,3,figsize=(16,200),sharey=True,sharex=True)
for i in range(x):
    subax0,subax1,subax2 = ax[i,0], ax[i,1], ax[i,2]
    subax0.imshow(np.abs(galIms[i].array),interpolation='none',vmax=vmax,norm=norm)
    _im2=subax1.imshow(np.abs(shearedFlat[i].array), interpolation='none', vmax=vmax,norm=norm)
    #_im2 = subax2.imshow(np.abs(shearedFlatNoE[i].array), interpolation='none', vmax=vmax,norm=norm)
    subax0.axes.axis('off');subax1.axes.axis('off');subax2.axes.axis('off')
    title = "RE={0} \n e1={1}, e2={2}".format(np.round(hlr[small100I[i]],3), np.round(e1c[i],3), np.round(e2c[i],3))
    title2 = "n={0} \n e1={1}, e2={2}".format(np.round(sn[small100I][i],3),np.round(e1[i],3), np.round(e2[i],3))    
    subax0.set_title(title2)
    subax1.set_title(title)
    #subax2.set_title(np.round(sn[small100I][i],3))
    if zoom:
        for subax in (subax0,subax1,subax2):
            subax.set_xlim(50,110)
            subax.set_ylim(50,110)
    fig.colorbar(_im2,ax=subax2)
#ax[0,0].set_title('Original')
#ax[0,1].set_title('Reconvolved')
#ax[0,2].set_title('Reconvolved_noDil')
plt.subplots_adjust(hspace=0.2,wspace=0.1)
name = '/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/convolutionEffects'
for i in range(len(flags)):
    if flags[i]:
        name += '-'+flagnames[i]
name+='.png'
plt.savefig(name)
print numDone,'done'


