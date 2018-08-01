import galsim
import numpy as np
import shape_measurement as sm
def measure_metacal(gals, psf, psfii, psf_galsample, pixel_scale, gal_oversample, shearList, gsparams, interpolant='lanczos100'):
    print "Number of gals to measure: ",len(gals)
    ii=0
    rece1,rece2 = [], []
    for gal in gals:
        ii+=1
        if ii%50==0: print ii
        fin = galsim.Convolve([gal,psf])
        given_im = fin.drawImage(scale=pixel_scale/gal_oversample, method='no_pixel')

        gal_interp = galsim.InterpolatedImage(given_im,gsparams=gsparams, x_interpolant=interpolant)
        shearedRecList = sm.cfGal(gal_interp, psfii, shearList)
        shearRece1, shearRece2 = [], []
        for rec in shearedRecList:
            recIm = rec.drawImage(scale=pixel_scale/gal_oversample,method='no_pixel')

            warningMessage = 'Warning: Shape measurement likely wrong, pixel scales differ'
            if given_im.scale != psf_galsample.scale or recIm.scale != psf_galsample.scale: print warningMessage

            #orig_shape= galsim.hsm.EstimateShear(given_im, psf_galsample, strict = False)
            rec_shape = galsim.hsm.EstimateShear(recIm,  psf_galsample, strict = False)

            if rec_shape.correction_status != 0: # or orig_shape.correction_status != 0 
                #orig_shape_e1, orig_shape_e2 = np.nan, np.nan
                rec_shape_e1, rec_shape_e2 = np.nan, np.nan
            else:
                #orig_shape_e1, orig_shape_e2 = orig_shape.corrected_e1, orig_shape.corrected_e2
                rec_shape_e1, rec_shape_e2 = rec_shape.corrected_e1, rec_shape.corrected_e2
            shearRece1.append(rec_shape_e1)
            shearRece2.append(rec_shape_e2)
        rece1.append(shearRece1); rece2.append(shearRece2)

    rece1 = np.array(rece1).transpose(); rece2 = np.array(rece2).transpose()
    return rece1, rece2



def measure_shearfirst(gals, psf, psfii, psf_galsample, pixel_scale, gal_oversample, shearList, gsparams, *args):
    ii=0
    print 'len gals: ',len(gals)
    e1arr, e2arr = [], []
    for gal in gals:
        ii+=1
        if ii%50==0: print ii
        shearGals = [gal.shear(shear) for shear in shearList]
        finArr = [galsim.Convolve([sgal,psf]) for sgal in shearGals]
        given_imArr = [fin.drawImage(scale=pixel_scale/gal_oversample, method='no_pixel') for fin in finArr]
        sheare1, sheare2 = [], []
        for given_im in given_imArr:
            if given_im.scale != psf_galsample.scale: print 'Warning: Shape measurement likely wrong, pixel scales differ'
            shape = galsim.hsm.EstimateShear(given_im,  psf_galsample, strict = False)

            if shape.correction_status != 0: 
                shape_e1, shape_e2 = np.nan, np.nan

            else:
                shape_e1, shape_e2 = shape.corrected_e1, shape.corrected_e2
            sheare1.append(shape_e1)
            sheare2.append(shape_e2)
        e1arr.append(sheare1); e2arr.append(sheare2)
    e1arr = np.array(e1arr).transpose(); e2arr = np.array(e2arr).transpose()
    return e1arr, e2arr


def rotGal(gal,intrinsicTheta, targetTheta):
    '''Rotate galaxy so major axis is rotated targetTheta clockwise relative to x axis'''
    if np.abs(targetTheta) > 2*np.pi:
        print 'warning: targetTheta > 2pi -- recall that targetTheta should be given in radians'
    angle = (targetTheta - intrinsicTheta)*galsim.radians
    return gal.rotate(angle)

def circularize(gal):
    inv=np.linalg.inv(gal.jac).flatten()
    newgal = gal.transform(*inv)
    return newgal

def makeGalaxy(cc, gal_indices, chromatic=False, gsparams=None, trunc_factor=0.):
    import scipy, scipy.special
    galaxies = [ ]
    sersicfit = cc.param_cat['sersicfit']
    for gal_id in gal_indices:
        gal_n = sersicfit[gal_id,2]
        gal_q = sersicfit[gal_id,3]
        gal_phi = sersicfit[gal_id,7]
        gal_hlr = sersicfit[gal_id,1]*np.sqrt(gal_q)*0.03 ## in arcsec
        b_n = 1.992*gal_n-0.3271
        gal_flux = 2*np.pi*gal_n*scipy.special.gamma(2*gal_n)*np.exp(b_n)*gal_q*(sersicfit[gal_id,1]**2)*(sersicfit[gal_id,0])/(b_n**(2.*gal_n))
        #print gal_n, gal_hlr, gal_flux, gal_q, gal_phi
        gal = galsim.Sersic(n=gal_n,half_light_radius=gal_hlr, flux=gal_flux, trunc=trunc_factor*gal_hlr, gsparams=gsparams).shear(q=gal_q,beta=gal_phi*galsim.radians)
        galaxies.append(gal)
    return galaxies
