""" Load data output by reconv_shear_metacal.py, calculate R and recover the cosmological shear, and plot.
Functions:
e2g: Convert from e-type to g-type, setting all |e| to nan.
bootstrap: Take averages of bootstrapped samples of R and ellipticity.
loadfiles: Load pkl files output by reconv_shear_metacal.py and make arrays.
calculateR: Calculate the R matrix from ellipticities.
matchNans: Ensure that the same galaxies are removed for all artificial shears and all image rotations (ie pairs)
recover_cosmoshear: Calculate recovered cosmological shear from R and ellipticity.
matchNansBetween: Match nans (logical or) between different sets of files (ie control and metacal branches)
main1: Filename to recovered shear pipeline for one set of files (ie just metacal/control, 1 lambda, etc)
main: Filename to recovered shear pipeline for several sets of files (eg control and metacal, 2 lambdas)
plot: Plot results (eg pixel size vs recovered shear)
"""

import numpy as np
from matplotlib import pyplot as plt
import galsim
import cPickle
import glob
import matplotlib
import re
import galFuncs

def e2g(e1, e2):
    """e1, e2 are floats; return g1, g2 or nan if |e|>1"""
    try:
        sh = galsim.Shear(e1=e1, e2=e2)
        return sh.g1, sh.g2
    except Exception as e:
        #warnings.warn (str(e))
        return np.nan, np.nan

def bootstrap(R, recg0, axisListRRand, axisListRMean, axisListRecg0Rand, axisListRecg0Mean, nBootstrap, noBootstrap=False):
    """ R: ndarray of R matrices, typically shape (2x2, 2 (rotations), ngal, nx^2, len(x)) where len(x) is the number of pixel sizes or similar
    recg0: ndarray of corrected g-type ellipticities with 0 artificial shear, typical shape (2 (g1,g2), 2 (rotations), ngal, nx^2, len(x))
    axisListRRand: List of ints, axes over which to randomize during bootstrap for R (eg (3, 4))
    axisListRMean: List of ints, axes over which to take the mean during bootstrap (eg (3, 4, 5), same as axisListRRand but including rotations)
    axisListRecg0Rand: Same as axisListRRand for recg0
    axisListRecg0Mean: Same as axisListRMean for recg0
    nBootstrap: int, Number of bootstrap samples to take
    noBootstrap: bool, default False, if False do bootstrapping, if True skip bootstrap and just take mean.
    """
    shape = np.array(R.shape)[axisListRRand]
    if noBootstrap:
        return np.array([np.nanmean(R, axis=tuple(axisListRMean))]), np.array([np.nanmean(recg0, axis=tuple(axisListRecg0Mean))])
    else:
        weights = galFuncs.getWeightArr(shape, nBootstrap)
        RBoot = galFuncs.bootstrapArr(R, axisListRRand, axisListRMean, weights)
        tmsBoot = galFuncs.bootstrapArr(recg0, axisListRecg0Rand, axisListRecg0Mean, weights)
        return RBoot, tmsBoot
        
def loadfiles(name, shape_method, killShearGT1):
    """Load files
    name: string pattern (ls style) giving the files you want.
    shape_method: string, "REGAUSS" or "KSB"
    killShearGT1: bool, True to remove shears greater than |e|>1 and False if not (relevant for KSB)
    Return
    recshape: ndarray of measured shapes, Shape (2 (g1,g2), #ArtificialShear, nrot (2), ngal, nx^2, len(x) (ie # pixel sizes))
    shearList, hlr, sn, q, phiList, ident: see reconv_shear_metacal
    """
    fn = sorted(glob.glob(name))
    nrot = len(fn)
    res = []
    ident0 = 0
    for i,filename in enumerate(fn):
        fil = open(filename)
        resmetacal = cPickle.load(fil)
        ident = resmetacal[-1]
        assert np.all(ident == np.roll(ident, 1, axis=0))
        assert (i == 0) | np.all(ident == ident0)
        ident0 = ident
        res.append(resmetacal)
        fil.close()
    res = np.stack(res).transpose()
    #res = [np.stack(x, axis=-1) for x in res]
    rece1, rece2, shearList, hlr, sn, q, phiList, ident = res
    hlr, sn, q, phiList, ident = [np.concatenate(x, axis=1) for x in [hlr, sn, q, phiList, ident]]
    rece1, rece2 = [np.concatenate(x, axis=2) for x in [rece1, rece2]]
    assert np.all([shearList[0] == shearList[i] for i in range(len(shearList))])
    shearList = shearList[0]
    # rece: nshear x pairs x ngal*nrot_nopair x nx^2
    # q, phiList, etc: pairs x ngal*nrot_nopair
    #nfiles = nrot_nopair

    # Size or other cuts
    smallMask = (hlr < np.inf)
    rece1[:, ~smallMask,:] = np.nan
    rece2[:, ~smallMask,:] = np.nan

    if shape_method == "REGAUSS":
        e2gVec = np.vectorize(e2g)
        recg1, recg2 = e2gVec(rece1, rece2)
    elif shape_method == "KSB":
        recg1, recg2 = rece1, rece2

    if killShearGT1:
        bad = recg1**2 + recg2**2 > 1
        recg1[bad] = np.nan
        recg2[bad] = np.nan

    # Combine g1, g2 into a single array
    recshape1, recshape2 = recg1, recg2
    recshape = np.stack([recshape1, recshape2])
    #(g1,g2), nshear, 2 (rotpairs), ngal, nx^2

    #Separate last (rotation / pixel size) axis or not
    sh = recshape.shape
    sepLastAx = True  # Set to False to average over x-axis (rotation, pixel size, etc)
    if not sepLastAx:
        recshape = recshape.reshape(sh+(1,))
    else:
        recshape = recshape.reshape(sh[:3]+(nrot, sh[3]/nrot,sh[4])) #Separate all rotations
        recshape = recshape.transpose(0,1,2,4,5,3)
    #print recshape.shape
    #((g1,g2), nshear, 2(rotpairs), ngal, nx^2, nrot)
    return recshape, shearList, hlr, sn, q, phiList, ident


def calculateR(shearArr, recshape, shearm_i1=1, shearp_i1=2, shearm_i2=3, shearp_i2=4):
    """ Calculate R for each galaxy image. R = deps_i / dg_j.
    shearArr: ndarray of floats, shape (nshear, 2 (g1,g2))
    recshape: see loadfiles
    shearm*: Indeces of shearminus and shearplus in the shearArr
    """
    # Calculate R for each galaxy image
    #R = deps_i / dg_j
    den = (shearArr[shearp_i1] - shearArr[shearm_i1] + shearArr[shearp_i2] - shearArr[shearm_i2])
    Ri1 = ((recshape[:, shearp_i1] - recshape[:, shearm_i1]).transpose() / den).transpose() ## dg1: R11 and R21
    Ri2 = ((recshape[:, shearp_i2] - recshape[:, shearm_i2]).transpose() / den).transpose() #dg2: R12 and R22
    R = np.stack([Ri1,Ri2]) #[[R11, R21], [R12, R22]]
    R = R.transpose([1,0] + range(2, len(R.shape))) #[[R11, R12], [R21, R22]]
    return R
    #mm = 0.5 * (R[0,0] + R[1,1]) - 1
    #R (2x2), rotpair (2), ngal, nx^2

def matchNans(R, recshape):
    # Make Mask: True if any R-component of any rotation of a galaxy is nan
    # shape ngal, nx^2
    galnanmask = np.isnan(R)
    galnanmask = np.any(galnanmask, axis=(0,1,2))

    # True if any shape measurement of any shear of a galaxy is nan (including 0)
    gnm2 = np.isnan(recshape)
    gnm2 = np.any(gnm2, axis=(0,1,2))
    # Combine the two masks
    galnanmask = galnanmask | gnm2

    ls = len(R.shape)
    transpose_arr = range(3,ls) + range(3) # (3,4,0,1,2) or (3,4,5,0,1,2)
    inv_transpose_arr = np.argsort(transpose_arr)
    #Make corresponding entry in other rotation nan if it is nan for one of the rotations
    #Make whole R matrix nan if one entry is nan
    #Match nans for all shears (including 0)
    newr = R.transpose(transpose_arr) # ngal, nx^2, R(2x2), nrot
    newr[galnanmask] = np.nan
    R = newr.transpose(inv_transpose_arr)

    #Match nans for shear / rotation
    #Match nans with R
    newrecshape = recshape.transpose(transpose_arr)
    newrecshape[galnanmask] = np.nan
    recshape = newrecshape.transpose(inv_transpose_arr)
    return R, recshape

def recover_cosmoshear(recshape, R, nBootstrap, noBootstrap=False):
    # Average over galaxies, subpixel shifts, rotations, and bootstrap
    axisListRRand, axisListRMean = [3,4], [2,3,4]
    axisListRecg0Rand, axisListRecg0Mean = [2,3], [1,2,3]
    #axisListRRTot, axisListRMTot = [3,4,5], [2,3,4,5]
    #axisListRecg0RTot, axisListRecg0MTot = [2,3,4], [1,2,3,4]

    RBoot,tmsBoot = bootstrap(R, recshape[:,0], axisListRRand, axisListRMean, axisListRecg0Rand, axisListRecg0Mean, nBootstrap, noBootstrap)
    RBoot, tmsBoot = RBoot.transpose(0,3,1,2), tmsBoot.transpose(0,2,1) # (nBootstrap, nrot, 2x2), (nBootstrap, nrot, g1_g2)

    rec_cosmoshear_boot = np.einsum('...jk,...k->...j', np.linalg.inv(RBoot), tmsBoot)
    rec_cosmoshear, rec_cosmoshear_std = np.mean(rec_cosmoshear_boot, axis=0), np.std(rec_cosmoshear_boot, axis=0)
    true_mean_shape, tmsstd = np.mean(tmsBoot, axis=0), np.std(tmsBoot, axis=0)

    cshear_err = rec_cosmoshear_std 
    tms_err = tmsstd 
    return rec_cosmoshear, rec_cosmoshear_std, true_mean_shape, tmsstd

def matchNansBetween(R, recshape):
    gnm_rcList = [np.isnan(rc) for rc in recshape]
    gnm_RList = [np.isnan(subR) for subR in R]
    gnm_rc, gnm_R = np.any(gnm_rcList, axis=0), np.any(gnm_RList, axis=0)
    recshape, R = np.stack(recshape), np.stack(R)
    for rc in recshape:
        rc[gnm_rc] = np.nan
    for RR in R:
        RR[gnm_R] = np.nan
    recshape = [recshape[i] for i in range(recshape.shape[0])]
    R = [R[i] for i in range(R.shape[0])]
    return R, recshape

def main1(name, shape_method, killShearGT1, nBootstrap, noBootstrap):
    recshape, shearList, hlr, sn, q, phiList, ident = loadfiles(name, shape_method, killShearGT1)
    shearArr = np.array([[sh.g1, sh.g2] for sh in shearList])
    R = calculateR(shearArr, recshape)
    R, recshape = matchNans(R, recshape)
    rec_cosmoshear, rec_cosmoshear_std, true_mean_shape, tmsstd = recover_cosmoshear(recshape, R, nBootstrap, noBootstrap)
    return rec_cosmoshear, rec_cosmoshear_std, true_mean_shape, tmsstd


def main(nameList, shape_method, killShearGT1, nBootstrap, noBootstrap):
    resList = [loadfiles(name, shape_method, killShearGT1) for name in nameList]
    res0 = resList[0]
    for res in resList:
        for val1, val0 in zip(res[1:], res0[1:]):
            assert np.all(val1 == val0)
    shearList = res0[1]
    shearArr = np.array([[sh.g1, sh.g2] for sh in shearList])
    recshapeList = [res[0] for res in resList]
    RList = [calculateR(shearArr, recshape) for recshape in recshapeList]
    Rrec = [matchNans(R, recshape) for R, recshape in zip(RList, recshapeList)]
    RList, recshapeList = [x[0] for x in Rrec], [x[1] for x in Rrec]
    RList, recshapeList = matchNansBetween(RList, recshapeList)
    res = [recover_cosmoshear(recshape, R, nBootstrap, noBootstrap) for recshape, R in zip(recshapeList, RList)]
    return res


def plot(resList, labelList, nameList, cshear):
    """ Plot results.
    reslist is the output of main, or main1 wrapped in a list
    labelList: list of strings, same length as resList, giving labels (eg ["metacal", "control"])
    nameList: Same as nameList passed to loadfiles, in order to load correct pixel scales
    cshear: Tuple of floats, giving cosmological shear used
   """
    xList=[]
    for name in nameList:
        fn = sorted(glob.glob(name))
        pixelscale = [float(re.findall("pixelscale0.\d\d?", x)[0][10:]) for x in fn]
        xList.append(pixelscale)

    fig,ax = plt.subplots(2,1,figsize=(6,8))
    for gindex in range(2):
        gis = str(gindex+1)
        subax=ax[gindex]
        for i, res in enumerate(resList):
            rec_cosmoshear, rec_cosmoshear_std = res[0], res[1]
            sorting = np.argsort(xList[i])
            subax.errorbar(np.array(xList[i])[sorting], rec_cosmoshear[:,gindex][sorting], rec_cosmoshear_std[:,gindex][sorting], label="Recovered g%s %s" % (gis, labelList[i]), fmt='o:')

        subax.axhline(cshear[gindex], c='k', linestyle=':')
        subax.legend()
        subax.set_xlabel("Pixel size (as)")
        subax.set_ylabel("g"+gis)
    ax[0].set_title("Shear %s" % str(cshear))
    plt.tight_layout()
    return fig, ax


###
""" Might call as follows
import numpy as np
import galsim
import responsivity
from matplotlib import pyplot as plt

shape_method = "REGAUSS"
killShearGT1 = False
nBootstrap = 1000
noBootstrap = False
procList = ["metacal", "control"]
shear="shear1"
sl = {"shear1":"0.02-0.0", "shear2":"0.0-0.02"}
sl2 = {"shear1": "0.02", "shear2": "0.02i"}
nameList = ["/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/redrawScaleFactor0.5/%s/*%s*.pkl" % (proc, sl[shear]) for proc in procList] # Load metacal and control

b = responsivity.main(nameList, shape_method, killShearGT1, nBootstrap, False)

fig, ax = responsivity.plot(b, procList, nameList, (0.02, 0))
ax[0].set_title("Shear 0.02, Regauss, Redraw Scale Factor 0.5")

"""
