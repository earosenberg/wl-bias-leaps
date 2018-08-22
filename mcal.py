"""
Implement metacal operations.
Functions:
dilate: Dilate a profile by 1+2g
dilatePSF: Dilate an analytic PSF (not interpolated image)
measureShapeBasic: Convolve gal with epsf and measure shape
measureShapeReconv: Convolve galaxy with epsf, draw image, interpolate, deconvolve, shear, reconvolve w/ dilated psf, measure shape.
galShiftErrs: Shift galaxy by sub-pixel shifts in a grid, optionally shear / metacal, measure shape.
galShiftErrsBatch: Perform galShiftErrs for several galaxies / shears.
"""

import galsim
import numpy as np

gsparams_default = galsim.GSParams(kvalue_accuracy=1.e-5, maximum_fft_size=2048*10, maxk_threshold=1.e-5)


def dilate(obj, shear):
    """Given profile object and shear object, return an object dilated by 1+2g"""
    dilation = 1.0 + 2.0*shear.g
    return obj.dilate(dilation)


def dilatePSF(epsf_profile, shear):
    """
    Separate epsf and pixel response, dilate by 1+2g, reconvolve by pixel response
    epsf_profile must be a Convolution object, not interpolated_image
    """
    obj_list = epsf_profile.obj_list
    non_pixel = [obj for obj in obj_list if not isinstance(obj, galsim.Pixel)]
    pixel = [obj for obj in obj_list if isinstance(obj, galsim.Pixel)][0]  # assume one pixel object
    if len(non_pixel) > 1:
        non_pixel = galsim.Convolution(non_pixel)
    else:
        non_pixel = non_pixel[0]
    dilation = 1.0 + 2.0 * shear.g
    dil_psf = non_pixel.dilate(dilation)
    reconv_psf = galsim.Convolution(dil_psf, pixel)
    return reconv_psf


def measureShapeBasic(gal, epsf, psfIm, ss, redrawScaleFactor=1., shear_est='REGAUSS', noise=None, noiseSNR=None, return_type='corrected', *args):
    """
    Convolve galaxy profile with epsf, measure shape.
    Parameters:
    gal:   galsim object representing the galaxy profile
    epsf:  galsim object representing the effective psf profile (w/ pixel response)
    psfIm: galsim Image of the epsf, *sampled at galaxy rate* (ss)
    ss:    float, sampling scale of the galaxy (pixel_scale/gal_oversample)
    redrawScaleFactor: float, factor by which to multiply ss when redrawing just before shape measurement
    shear_est: string, default "REGAUSS", the shear estimator to use in shape measurement. Can be "REGAUSS" or "KSB".
    noise: Galsim noise object or None, default None; If not None, the generator of noise to be added to each image.
    noiseSNR: float or None, default None; If not None, the SNR of the galaxy after adding noise at constant flux.
    return_type: string, default "corrected", can be "image", "observed", or "corrected"

    Returns:
    (e1, e2) if shear_est=="REGAUSS" and return_type=="corrected" or "observed"
    (g1, g2) if shear_est=="KSB" and return_type=="corrected" or "observed"
    (nan, nan) in the above two cases if shape measurement fails
    (Im, Im) where Im is the galaxy image after psf convolution and noise addition, just prior to shape measurement, if return_type=="image"
    """
    # Convolve galaxy and epsf, and draw image
    fin = galsim.Convolve(gal, epsf)
    galIm = fin.drawImage(scale=ss*redrawScaleFactor, method='no_pixel')

    # Add Noise
    if noise is not None:
        if noiseSNR is not None:
            galIm.addNoiseNSR(noise, snr=noiseSNR, preserve_flux=True)
        else:
            galIm.addNoise(noise)

    # Measure shape and return appropriate quantity depending on input parameters
    warningMessage = 'Warning: Shape measurement likely wrong, pixel scales differ'
    if galIm.scale != psfIm.scale: print warningMessage, "galIm.scale: ", galIm.scale, "psfIm.scale: ", psfIm.scale
    shape = galsim.hsm.EstimateShear(galIm, psfIm, strict=False, shear_est=shear_est)
    if return_type == 'image':
        return galIm, galIm
    elif return_type == 'observed':
        if shape.moments_status != 0:
            return np.nan, np.nan
        else:
            if shear_est == "REGAUSS":
                return shape.observed_shape.e1, shape.observed_shape.e2
            elif shear_est == "KSB":
                return shape.observed_shape.g1, shape.observed_shape.g2
            else:
                raise ValueError("Only REGAUSS and KSB are supported values of shear_est; not shear_est="+shear_est)
    else:
        if shape.correction_status != 0:
            return np.nan, np.nan
        else:
            if shear_est == "REGAUSS":
                return shape.corrected_e1, shape.corrected_e2
            elif shear_est == "KSB":
                return shape.corrected_g1, shape.corrected_g2
            else:
                raise ValueError("Only REGAUSS and KSB are supported values of shear_est; not shear_est="+shear_est)

def measureShapeControl(gal, epsf, psfIm, ss, redrawScaleFactor=1., shear_est="REGAUSS", noise=None, noiseSNR=None, return_type='corrected', artificialShear=galsim.Shear(g1=0.,g2=0.), *args):
    dil_psf = dilatePSF(epsf, artificialShear)
    shear_gal = gal.shear(g1=artificialShear.g1, g2=artificialShear.g2)
    return measureShapeBasic(shear_gal, dil_psf, psfIm, ss, redrawScaleFactor, shear_est, noise, noiseSNR, return_type)

def measureShapeReconv(gal, epsf, psf_galsample, galscale, redrawScaleFactor, shear_est, noise, noiseSNR,
                       return_type, artificialShear, psfii,
                       interpolant='lanczos100', gsparams=gsparams_default):
    """
    Convolve galaxy with epsf, draw image, interpolate, deconvolve by psfii, shear, reconvolve by dilated psf, measure shape.
    Parameters:
    gal:   galsim object representing the galaxy profile
    epsf:  galsim object representing the effective psf profile (w/ pixel response)
    psf_galsample: galsim Image of the **DILATED** epsf that you will reconvolve by, *sampled at galaxy rate* (ss)
    galscale: float, sampling scale of the galaxy (pixel_scale/gal_oversample)
    shear_est: string, the shear estimator to use in shape measurement. Can be "REGAUSS" or "KSB".
    noise: Galsim noise object or None; If not None, the generator of noise to be added to each image.
    noiseSNR: float or None; If not None, the SNR of the galaxy after adding noise at constant flux.
    return_type: string, can be "image", "observed", or "corrected"
    artificialShear: galsim Shear object, the artificial shear to be applied to the galaxy.
    psfii: interpolated image of the oversampled psf
    interpolant: string, default 'lanczos100'
    gsparams: Galsim gsparams, default gsparams_default (given at beginning of this file).

    Convolve gal+epsf profiles, create interpolated image, de/reconvolve by psfii, measure shape
    Returns either the image or the observed or corrected (e1, e2) or (g1, g2) of reconvolved image
    See measureShapeBasic for full description of return signature
    """
    fin = galsim.Convolve(gal, epsf)
    given_im = fin.drawImage(scale=galscale, method='no_pixel')
    if noise is not None:
        if noiseSNR is not None:
            given_im.addNoiseSNR(noise, snr=noiseSNR, preserve_flux=True)
        else:
            given_im.addNoise(noise)
    gal_interp = galsim.InterpolatedImage(given_im, gsparams=gsparams, x_interpolant=interpolant)
    inv_psf = galsim.Deconvolve(psfii)
    dec = galsim.Convolve(gal_interp, inv_psf)
    if interpolant == "sinc" and artificialShear == galsim.Shear(g1=0, g2=0):
        # Breaks if sinc, artificialShear=0, and try to reconvolve with analytic psf
        raise Exception("Due to possible bug, cannot use sinc/zero shear/non-interpolated psf to reconvolve")
    dil_psf = dilatePSF(epsf, artificialShear)  # CHANGE
    shear_gal = dec.shear(g1=artificialShear.g1, g2=artificialShear.g2)
    return measureShapeBasic(shear_gal, dil_psf, psf_galsample, galscale, redrawScaleFactor, shear_est, None, None, return_type)


def galShiftErrs(gal, epsf, epsf_galsample, nx, ss, redrawScaleFactor=1., shear_est='REGAUSS', noise=None, noiseSNR=None, measureShape=measureShapeBasic, psfii=None,
                 artificialShear=galsim.shear.Shear(g1=0., g2=0.), interpolant='lanczos100',
                 return_type='corrected', gsparams=gsparams_default):
    """
       Shift galaxy by sub-pixel shifts in a grid, optionally shear / metacal, measure shape.
       Parameters:
       gal, epsf: galsim objects representing galaxy / effective psf profiles
       epsf_galsample: Image of the DILATED epsf drawn at ss, so it doesn't have to be re-drawn every time
       nx: int, number of steps with which to shift galaxy eg nx=5 shifts by 0, 1/8=1/(2*(nx-1)), 2/8, 3/8, 4/8
       ss: galaxy sampling rate; galaxies will be shifted up to (ss/2, ss/2) in each dimension
       Optional arguments shear_est="REGAUSS", noise=None, noiseSNR=None, psfii, artificialShear, interpolant, return_type, gsparams as in measureShapeBasic/measureShapeReconv
       measureShape: function that takes gal, epsf, psfIm, ss, possibly more args, returns (e1, e2); default measureShapeBasic
       psfii: Optional interpolated image of psf, for measureShapeReconv
       Return arrays of e1, e2 measured for each image of a galaxy shifted in an nx x nx grid between (0,0), (ss/2, ss/2)
       inclusive. Psf is not shifted.
       Return (e1arr, e2arr) where each is a list of length nx^2
    """
    d1shifts = np.linspace(0, ss/2, nx)
    aa, bb = np.meshgrid(d1shifts, d1shifts)
    d2shifts = zip(aa.flatten(), bb.flatten())

    e1arr, e2arr = [], []
    for shift in d2shifts:
        galshift = gal.shift(shift)
        e1, e2 = measureShape(galshift, epsf, epsf_galsample, ss, redrawScaleFactor, shear_est, noise, noiseSNR, return_type, artificialShear, psfii, interpolant, gsparams)
        e1arr.append(e1); e2arr.append(e2)
    # print 'Num failures: %d / %d' % (np.sum(np.isnan(e1arr)), len(e1arr))
    return e1arr, e2arr


# This function is just vectorized to take a galaxy array of arbitrary shape
# mostly maintained for compatibility with old notebooks, but occasionally useful
galShiftErrsVec = np.vectorize(galShiftErrs, otypes=[object, object],
                               excluded=set([1, 2, 3, 4, 'redrawScaleFactor', 'measureShape', 'psfii',
                                             'artificialShear', 'interpolant',
                                             'return_type', 'shear_est', 'gsparams']))


def galShiftErrsBatch(galList, epsf, nx, ss, redrawScaleFactor=1., shear_est='REGAUSS', noise=None, noiseSNR=None,
                      measureShape=measureShapeBasic, psfii=None, artificialShearList=[galsim.Shear(0j)],
                      interpolant='lanczos100', return_type='corrected', gsparams=gsparams_default):
    """
       Wrap galShiftErrs to take many galaxies and shears.
       Arguments same as galShiftErrs except for galList and artificialShearList which are lists of galaxy profile and shear objects.
       Return (e1arr, e2arr) where each has shape (shear x ngal x nx^2).
    """
    lst1, lst2 = [], []
    print 'Num shears: ', len(artificialShearList)
    for shear in artificialShearList:
        gl1, gl2 = [], []
        dil_psf = dilatePSF(epsf, shear)
        dil_psfIm = dil_psf.drawImage(scale=ss*redrawScaleFactor, method='no_pixel')
        counter = 0
        print 'Num gals: ', len(galList)
        for gal in galList:
            if counter % 50 == 0:
                print counter
            counter += 1
            se = galShiftErrs(gal, epsf, dil_psfIm, nx, ss, redrawScaleFactor, shear_est, noise, noiseSNR,
                              measureShape, psfii, shear, interpolant, return_type, gsparams)
            gl1.append(se[0]); gl2.append(se[1])
        lst1.append(gl1); lst2.append(gl2)
    e1arr, e2arr = np.array(lst1), np.array(lst2)
    return e1arr, e2arr
