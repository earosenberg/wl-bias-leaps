""" 
This is the primary script used to run the metacal process.
Functions:
removeBulge -- Given a galaxy profile, return the disk part if it is a disk+bulge. Else return the given profile.
makeSersic -- Return a Sersic galaxy profile with given n, circularized hlr, flux, q, phi. Used for grids.
makeGalsGrid -- Given a single sersic n and angle phi, and distributions in hlr and q, make Sersic profiles for the hlr x q grid.
makeGalsComos -- Make ngal galaxy profiles taken from the catalog.
main -- Create a set of galaxies, optionally shear, optionally perform metacal process, measure shape.
"""
import numpy as np
import galsim
import mcal
import galFuncs
import cPickle
import sys

default_gsparams = galsim.GSParams(kvalue_accuracy=1.e-5, maximum_fft_size=2048*10,
                                   maxk_threshold=9.e-4)


def removeBulge(gal):
    """Given a galaxy profile gal, return the disk part if it is a sum; else return it unchanged."""
    if type(gal) != galsim.compound.Sum:
        return gal
    else:
        disk = gal.obj_list[1]
        return disk


def makeSersic(gal_n, gal_hlr, gal_flux, gal_q, gal_phi, gsparams):
    """Return a Sersic profile with given n, circ. hlr, flux, axis ratio q, orientation angle phi in radians."""
    gal = galsim.Sersic(n=gal_n, half_light_radius=gal_hlr, flux=gal_flux,
                        gsparams=gsparams).shear(q=gal_q, beta=gal_phi*galsim.radians)
    return gal


def makeGalsGrid(gsparams, gal_phi, gal_n, circ_hlr_dist, q_dist):
    """ Make an array of galaxy profiles in a grid of circ. hlr and q.
    Parameters:
    gsparams -- gsparams of the galaxy
    gal_phi, gal_n -- floats giving the orientation angle phi in radians and Sersic n (0.3<=n<=6.2)
    circ_hlr_dist, q_dist -- 1d arrays, giving circularized hlr in arcsec and q
    Returns:
    gals -- 1d array of galaxy profiles of length len(circ_hlr_dist) x len(q_dist)
    hlr, sn, q, phi -- 1d array giving the corresponding quantity, same length as gals
    ident -- range(len(hlr))
    """
    gal_flux = 100.
    gals = np.array([makeSersic(gal_n, circ_hlr, gal_flux, q, gal_phi, gsparams)
                     for q in q_dist for circ_hlr in circ_hlr_dist])
    hlrList = np.array([circhlr for q in q_dist for circhlr in circ_hlr_dist])
    qList = np.array([q for q in q_dist for hlr in circ_hlr_dist])
    hlr, sn, q, phi = hlrList, np.ones_like(hlrList) * gal_n, qList, np.ones_like(hlrList) * gal_phi
    ident = range(len(hlrList))
    return gals, hlr, sn, q, phi, ident


def makeGalsCosmos(gsparams, ngal, start=0, stop=None):
    """ Make an array of galaxy profiles taken from the COSMOS catalog.
    Note that cuts on galaxies are hard coded in this function.
    Parameters: 
    gsparams -- gsparams of the galaxy
    ngal -- Int, number of galaxy profiles to produce (NOT RETURN)
    start -- Int, default 0. Index in the len(ngal) galaxy list at which to start returning
    stop -- Int or None, default None. Index at which to stop returning. Start and Stop are used to parrallelize a run of ngal galaxies into smaller chunks.
    Returns:
    gals, hlr, sn, q, phi, ident -- Same as makeGalsGrid. gals is len(stop-start)
    """
    # Load galaxy catalog and select galaxies
    cc = galsim.COSMOSCatalog(dir='/disks/shear15/KiDS/ImSim/pipeline/data/COSMOS_25.2_training_sample/', use_real=False)
    paramrecIndeces = []
    # Make list of indeces of galaxies that satisfy selection criteria
    for i in xrange(cc.getNObjects()):
        paramrec = cc.getParametricRecord(i)
        paramrecsersic = paramrec['sersicfit']
        hlr, sn, q = [paramrecsersic[j] for j in (1, 2, 3)]
        # Choose large galaxies, reasonable sersic n
        if hlr * np.sqrt(q) * 0.03 > 0.05 and sn >= 0.3:  # and paramrec['use_bulgefit']==1
            paramrecIndeces.append(i)
        if len(paramrecIndeces) >= ngal:
            break
    paramrecIndeces = paramrecIndeces[start:stop]
    # Make galaxy profiles
    gals = cc.makeGalaxy(paramrecIndeces, chromatic=False, gsparams=gsparams)  # , sersic_prec=0.5)
    # Load parameters of the newly made galaxies and put into lists
    # For bulge+disk profiles the sersic parameters corresponding to the disk part are saved
    paramrec = cc.getParametricRecord(paramrecIndeces)
    corr_sersicfit = paramrec['sersicfit']
    corr_bulgefit = paramrec['bulgefit'][:, :8]
    corr_usebulgefit = paramrec['use_bulgefit']
    corr_usebulgefit = corr_usebulgefit.reshape(len(corr_usebulgefit), 1)
    corr_usebulgefit = np.repeat(corr_usebulgefit, 8, axis=1)
    use_sers = (1+corr_usebulgefit) % 2
    corr_fit_params = corr_sersicfit * use_sers + corr_bulgefit*corr_usebulgefit

    hlr, sn, q, phi = [corr_fit_params[:, i] for i in (1, 2, 3, 7)]
    ident = cc.getParametricRecord(paramrecIndeces)['IDENT']
    hlr = hlr * np.sqrt(q) * 0.03  # convert to circularized hlr in as
    return gals, hlr, sn, q, phi, ident


def main(gridparams, ngal, shearList, cosmo_shear, method, measurement, rotGals, rotAngleDeg, pixel_scale=0.1, nrot=1,
         lamda=800., diameter=1.2, psf_oversample=5., gal_oversample=1., nx=5, interpolant='lanczos100', redrawScaleFactor=1.,
         circ=False, bulgeHandling=1, noise=None, noiseSNR=None, gsparams=default_gsparams, save=True, start=0, stop=None, shear_est='REGAUSS'):
    """ 
    main -- Create a set of galaxies, optionally shear, convolve w/ obscured Airy psf, optionally perform metacal process, measure shape.
    Parameters:
    gridparams -- List containing [gal_n, circ_hlr_dist, q_dist] as required by makeGalsGrid. None if makeGalsCosmos is being used.
    ngal -- Int containing total number of galaxies of the run, IF makeGalsCosmos is being used. None if makeGalsGrid is being used.
    shearList -- List of galsim.Shear objects giving artificial shears to apply if metacal is performed (measurement==mcal.measureShapeReconv)
    cosmo_shear -- galsim.Shear object, the cosmological shear to be applied to all galaxies before PSF convolution
    method -- string, the name of the measurement method for the savename
    measurement -- function, the processing method to be used. Expects either mcal.measureShapeReconv, mcal.measureShapeBasic, or mcal.measureShapeControl
    rotGals -- bool. True means galaxies should be rotated to a given angle immediately after production. False leaves intrinsic rotation. True for makeGalsGrid.
    rotAngleDeg -- Orientation angle in degrees clockwise of x-axis that all galaxies should have if rotGals is True
    pixel_scale -- float, default=0.1, pixel size in arcsec
    nrot -- int>=1, default=1. Number of rotations for each galaxy. 1 is just original angle, 2 is (orig, orig+90deg), 3 (0, 60, 120), etc
    lamda -- float, default=800., wavelength for Airy PSF in nm
    diameter -- float, default=1.2, diameter of aperture for Airy PSF, in m
    psf_oversample -- float, default=5., factor by which to oversample psf profile, such that psf interpolated image is taken from drawing at pixel scale pixel_scale/psf_oversample.
    gal_oversample -- float, defaule=1., factor by which to oversample galaxy profile, galaxy interpolated image is taken from drawing at pixel scale pixel_scale/gal_oversample.
    nx -- int>=1, default=5, Number of subpixel shifts to perform for each galaxy in each dimension. 
    interpolant -- string, default='lanczos100', interpolant to use.
    circ -- bool, default=False, if True circularize all galaxy profiles (currently broken unless bulge removal is re-implemented)
    bulgeHandling -- int, selects option of how to handle bulge+disk. Currently broken, bulge+disk always left in.
    noise -- galsim Noise object or None, the noise to add to each frame (note that the noise realization will be *different* for each one)
    noiseSNR -- float or None. If None and noise is not None, noise is added at sigma in noise object. If both noise and noiseSNR are not None, noise is added to make SNR noiseSNR, at constant flux.
    gsparams -- gsparams of galaxies
    save -- bool, default True, save a pkl of the results before returning if True.
    start -- int, default 0, start parameter to be passed to makeGalsCosmos
    stop -- int or None, default None, stop parameter to be passed to makeGalsCosmos
    shear_est -- string, default "REGAUSS", the shape estimator to be used. Can be "REGAUSS" or "KSB".
    Returns: 
    res -- list, output from measurement function. [e1arr, e2arr, shearList, hlr, sn, q, phi, ident], further documentation in mcal.py
    Other Effects:
    if save is True, saves res in a pkl file that has a long name based on many (but not all!!) of the input parameters
    
    """
    rotAngleRad = np.radians(rotAngleDeg)
    # Make psf
    airy = galsim.Airy(lam=lamda, diam=diameter, scale_unit=galsim.arcsec,
                       obscuration=0.3, gsparams=gsparams)
    pixel = galsim.Pixel(pixel_scale, gsparams=gsparams)
    psf = galsim.Convolve(airy, pixel)
    given_psf = psf.drawImage(scale=pixel_scale/psf_oversample, method='no_pixel')
    # Draw PSF at galaxy scale for shape measurement
    # psf_galsample = psf.drawImage(scale=pixel_scale/gal_oversample, method='no_pixel')
    psfii = galsim.InterpolatedImage(given_psf, gsparams=gsparams, x_interpolant=interpolant)

    # Make galaxy profiles
    print "Making gals..."
    if gridparams is None:
        gals, hlr, sn, q, phi, ident = makeGalsCosmos(gsparams, ngal, start, stop)
    else:
        gals, hlr, sn, q, phi, ident = makeGalsGrid(gsparams, rotAngleRad, *gridparams)
        ngal = len(ident)
    print "Done"
    
    # Option to circularize galaxies (BROKEN)
    if circ:
        gals = [galFuncs.circularize(gal) for gal in gals]
        q = np.ones_like(sn)
    # Option to rotate galaxies to a particular orientation wrt coordinate axes -- will not work on bulge+disk
    if rotGals:
        gals = [galFuncs.rotGal(gal, intrinsicAngle, rotAngleRad)
                for gal, intrinsicAngle in zip(gals, phi)]
        phi = np.ones_like(phi) * rotAngleRad

    # Generate rotated galaxies, add to galaxy list
    rotation_angle = [180./(nrot)*i for i in range(1, nrot)]
    rot_gals = [gal.rotate(angle*galsim.degrees) for angle in rotation_angle for gal in gals]
    gals = np.concatenate((gals, rot_gals))
    hlr = np.stack([hlr]*nrot)
    sn = np.stack([sn]*nrot)
    q = np.stack([q]*nrot)
    newphi = [subphi + np.radians(newrot) for newrot in rotation_angle for subphi in phi]
    tmpphi = np.concatenate((phi, newphi))
    phi = tmpphi.reshape(nrot, phi.shape[0])
    ident = np.stack([ident] * nrot)
    # Add cosmological shear
    gals = [gal.shear(cosmo_shear) for gal in gals]
    print "Beginning measurements"
    # Measure e1, e2

    e1arr, e2arr = mcal.galShiftErrsBatch(gals, psf, nx, pixel_scale/gal_oversample, redrawScaleFactor, shear_est, noise, noiseSNR, measurement, psfii, shearList, interpolant, 'corrected', gsparams) 
    e1arr = e1arr.reshape(e1arr.shape[0], nrot, e1arr.shape[1]/nrot, e1arr.shape[2])
    e2arr = e2arr.reshape(e2arr.shape[0], nrot, e2arr.shape[1]/nrot, e2arr.shape[2])
    # e1arr and e2arr are (nShear, nrot, ngal, nx^2)
    # hlr, sn, q, phi, ident are (nrot, ngal)
    res = [e1arr, e2arr, shearList, hlr, sn, q, phi, ident]  # , gals]

    if save:
        # Prepare to save data
        basedir = '/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/'
        basename = basedir + 'measErrs_diskonly'
        if gridparams is None:
            basename = basename + '_cosmos'
            if stop is not None:
                basename = basename+"_start%04d" % start
        else:
            basename = basename + '_grid_n%s' % gridparams[0]
        if circ:
            basename = basename + '_circ'
        if rotGals:
            basename = basename+'_rot'+str(rotAngleDeg)+'deg'
        name = basename + '_lambda%d' % (lamda)
        if len(shearList) == 1:
            shearname = str((shearList[0].g1, shearList[0].g2))
            shearname = shearname.replace(', ', '-')[1:-1]
        else:
            shearname = len(shearList)

        cosmoshearname = str((cosmo_shear.g1, cosmo_shear.g2))
        cosmoshearname = cosmoshearname.replace(', ', '-')[1:-1]
        name = name + '_shear%s_cosmoshear%s' % (shearname, cosmoshearname)
        name = name + '_'+shear_est
        if noise is not None:
            if noiseSNR is not None:
                name = name + '_noiseSNR%s' % noiseSNR
            else:
                name = name + '_noiseSTD%s' % noise.sigma
        savename = name + '_%dgals_%s_%s_nx%s_nrot%s_pixelscale%s._redrawSF%s.pkl' % (ngal, method, interpolant, nx, nrot, pixel_scale, redrawScaleFactor)

        # Save data
        fil = open(savename, 'wb')
        cPickle.dump(res, fil)
        fil.close()
        print 'Saved %s' % savename
    return res

# Grid parameters
gal_n = 1.0  # 0.5, 1.0, 4
circ_hlr_dist = np.linspace(0.05, 1.0, 2)
q_dist = np.linspace(0.1, 1, 10)
gridparams = [gal_n, circ_hlr_dist, q_dist]
# gridparamstemp = [gal_n, np.linspace(0.5,1.0,2), np.linspace(0.5,0.75,2)] #Small grid for tests

# COSMOS parameters
ngal = 1000
rotGals = False
start = 0
stop = None

# Make shearList
sstep = 0.01
sl = [(0., 0.), (-sstep, 0.), (sstep, 0.), (0., -sstep), (0., sstep)]
shearList = [galsim.shear.Shear(g1=sh[0], g2=sh[1]) for sh in sl]

# Both parameters
rotAngleDeg = 0  # Degrees
methodI = 2  # Choose measurement and method -- should be 1 or 2 -- 1 for metacal, 2 for control
nrot = 2
methods = ['shearfirst', 'metacalshear', 'control']
measurements = [mcal.measureShapeBasic, mcal.measureShapeReconv, mcal.measureShapeControl]
method, measurement = methods[methodI], measurements[methodI]
shear_est = "REGAUSS" #"KSB"
cosmo_shear = galsim.shear.Shear(g1=0.0, g2=0.01)
#cosmo_shear = galsim.shear.Shear(g1=0.01, g2=0.0)
noiseSNR = None
noise = None
#pixel_scale=0.07

# Make noise
# noiseSNR = 50
# noiseSigma = 1
# seed = 1234 #CAUTION: Different rotations/pixel sizes started in different scripts will have same seed
# rng = galsim.BaseDeviate(seed=seed)
# noise = galsim.GaussianNoise(sigma=noiseSigma, rng=rng)

# options #
# circ = False
# bulgeHandling = 0  # Only sersic implemented

#For parallelizing: number of galaxies to make in each sub-run
#numGalStep = 250
# ii = 1
# start, stop = ii*numGalStep, (ii+1)*numGalStep

if __name__ == '__main__':
    #Get arguments from the command line for batch processing. I just vary one of these at a time, so I comment out the others
    cm_args = sys.argv
    if len(cm_args) == 2:
        # Batch number for parallelization with COSMOS
        # ii = int(cm_args[1])
        # start, stop = ii*numGalStep, (ii+1)*numGalStep

        # Rotation angle for grid
        # rotAngleDeg = int(cm_args[1])
        # print 'Rotation Angle: %s degrees' % rotAngleDeg

        # pixel scale
        pixel_scale = float(cm_args[1])
        print "pixel scale: %.2g" % pixel_scale

        # Noise
        # if cm_args[1] == 'None':
            #     noiseSNR = None
            # else:
            #     noiseSNR = int(cm_args[1])
            # print 'noiseSNR: %s' % noiseSNR
    elif len(cm_args > 2):
        raise Exception("More than one command line argument")
#    res = main(None, ngal, shearList, cosmo_shear, method, measurement, rotGals, rotAngleDeg, nx=5, nrot=nrot, interpolant='lanczos200', pixel_scale=pixel_scale)  # Cosmos
    if noiseSNR is None:
        noise = None

    # Run with COSMOS
    main(None, ngal, shearList, cosmo_shear, method, measurement, rotGals, rotAngleDeg, nx=5, nrot=nrot, interpolant='lanczos100', pixel_scale=pixel_scale, start=start, stop=stop, lamda=800., shear_est=shear_est, redrawScaleFactor=1.0)
  # Run with grid
  # res = main(gridparams, None, shearList, cosmo_shear, method, measurement, True, rotAngleDeg, nx=1, nrot=nrot, interpolant='lanczos100', noise=noise, noiseSNR=noiseSNR, pixel_scale=0.1)  # Grid
