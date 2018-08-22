""" Miscellaneous functions operating on galaxies or galaxy data, applied in mcal.py and reconv_shear_metacal.py
Fucntions:
rotGal: Rotate a galaxy to a given orientation relative to the x-axis. Currently does NOT WORK on bulge+disk galaxies.
circularize: Undo all shears applied to a galaxy profile, circularizing it. Does NOT WORK on bulge+disk.
makeGalaxy: Make Sersic galaxies from the COSMOS catalog
nanav: Take a weighted average of an array along a given set of axes, ignoring nans
getRandomWeights: Make an array of random weights for use in bootstrapping.
matchShapes: Reshape array1 whose shape is a subset of array2 to match shape of array2 by repeating axes
getWeightArr: Make an array of nBootstrap arrays of random weights
bootstrapArr: Take a weighted mean along given axis, with weights, to get bootstrapped samples
"""

import warnings
import numpy as np
import galsim
import scipy
import scipy.special

def rotGal(gal, intrinsicTheta, targetTheta):
    """Rotate galaxy so major axis is rotated targetTheta radians clockwise relative to x axis"""
    if np.abs(targetTheta) > 2*np.pi:
        print 'warning: targetTheta > 2pi -- recall that targetTheta should be given in radians'
    angle = (targetTheta - intrinsicTheta)*galsim.radians
    return gal.rotate(angle)


def circularize(gal):
    """Undo all shears applied to a galaxy and return the result"""
    inv = np.linalg.inv(gal.jac).flatten()
    newgal = gal.transform(*inv)
    return newgal


def makeGalaxy(cc, gal_indices, chromatic=False, gsparams=None, trunc_factor=0.):
    """Return Sersic profile specified by Cosmos Catalog (cc) and gal_indeces"""
    galaxies = []
    sersicfit = cc.param_cat['sersicfit']
    for gal_id in gal_indices:
        gal_n = sersicfit[gal_id, 2]
        gal_q = sersicfit[gal_id, 3]
        gal_phi = sersicfit[gal_id, 7]
        gal_hlr = sersicfit[gal_id, 1] * np.sqrt(gal_q) * 0.03  # in arcsec
        b_n = 1.992 * gal_n - 0.3271
        gal_flux = 2 * np.pi * gal_n * scipy.special.gamma(2*gal_n)*\
                   np.exp(b_n) * gal_q * (sersicfit[gal_id, 1]**2)*\
                   (sersicfit[gal_id, 0])/(b_n**(2. * gal_n))
        # print gal_n, gal_hlr, gal_flux, gal_q, gal_phi
        gal = galsim.Sersic(n=gal_n, half_light_radius=gal_hlr, flux=gal_flux,
                            trunc=trunc_factor*gal_hlr, gsparams=gsparams).shear(
                                q=gal_q, beta=gal_phi*galsim.radians)
        galaxies.append(gal)
    return galaxies

### Functions applied primarily to bootstrapping

def nanav(arr, weights, axis=None):
    '''Take weighted average of an array, ignoring nan values
       array: ndarray object. weights: array of same size, giving weights.
       axis: int giving axis to average along
       returns: array averaged along given axis, ignoring nans'''
    marr = np.ma.MaskedArray(arr, mask=np.isnan(arr))
    av = np.ma.average(marr, weights=weights, axis=axis)
    if isinstance(av, np.ndarray):
        if np.any(av.mask):
            warnings.warn("Average should not contain nan")
            return av.data
            #raise ValueError, 'Average should not contain nan'
        else:
            return av.data
    else:
        return av

def getRandomWeights(size, shape=None, seed=None):
    """Make a list of size <<size>> of random weights, optionally reshape to <<shape>>."""
    np.random.seed(seed=seed)
    randNums = np.random.random(size)
    ncounts, bin_edges = np.histogram(randNums, bins=size, range=(0,1))
    if shape is not None:
        return ncounts.reshape(shape)
    else:
        return ncounts

def matchShapes(arr1, arr2, axisList):
    """ Make arr1 and arr2 have the same shapes.
    arr1 is the array whose shape you want both to share, arr2 has a shape that is a subset of arr1's
    axisList is a list of ints giving the dimensions of arr1 that should be added to arr2.
    Ex: arr1 has shape (2,4,5,1,3) and arr2 (4,5,3), axisList = [0,3]
    Then arr2 is repeated along each axis in axisList (here 0, 3) until it has the same shape as arr1. 
    This expanded arr2 is returned.
    """
    if len(arr1.shape) < len(arr2.shape):
        arr1, arr2 = arr2, arr1
    sh1, sh2 = arr1.shape, arr2.shape
    if len(sh1) != len(sh2):
        newshape = [sh1[i] if i in axisList else 1 for i in range(len(sh1))]
        arr2 = arr2.reshape(newshape)
        sh2 = arr2.shape
    for i in range(len(sh1)):
        if sh1[i] != sh2[i]:
            arr2 = np.repeat(arr2, sh1[i], axis=i)
    return arr2

def getWeightArr(shape, nBootstrap):
    """ Get nBootstrap sets of random weights and return an ndarray of shape (nBootstrap, shape). """
    size = np.product(shape)
    btList = [getRandomWeights(size, shape) for _ in range(nBootstrap)]
    return np.stack(btList)


def bootstrapArr(arr, axisListMatch, axisListMean, weightArr):
    """ Match weight and array shapes, then bootstrap and return the nanmean along given axes.
    arr: The array whose bootstrapped mean you want.
    axisListMatch: list of ints, The list of dimensions to be passed to matchShapes
    axisListMean: list of ints, giving the dimensions over which a mean should be taken
    weightArr: Array of random weights of shape (nBootstrap, sh) where sh is a subset of arr.shape. Get from getWeightArr.
    Return an array of (nBootstrap, sh2) where sh2 is arr.shape - axisListMean
    """
    btList = []
    for weights in weightArr:
        weights = matchShapes(arr, weights, axisListMatch)
        btList.append(nanav(arr, weights, tuple(axisListMean)))
    return np.stack(btList, axis=0)

###
