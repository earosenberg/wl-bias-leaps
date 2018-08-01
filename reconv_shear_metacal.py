# Possible next step: command line arguments
import numpy as np
import galsim
import cPickle
import mcal

default_gsparams = galsim.GSParams(kvalue_accuracy=1.e-5, maximum_fft_size=2048*10,
                                   maxk_threshold=9.e-4)


def removeBulge(gal):
    if type(gal) != galsim.compound.Sum:
        return gal
    else:
        disk = gal.obj_list[1]
        return disk

def main(ngal, shearList, method, measurement, rotGals, rotAngleDeg, pixel_scale=0.1,
         lamda=800., diameter=1.2, psf_oversample=5, gal_oversample=1, interpolant='lanczos100',
         circ=False, bulgeHandling=1, gsparams=default_gsparams, save=True):
    rotAngleRad = np.radians(rotAngleDeg)
    # #Make psf##
    airy = galsim.Airy(lam=lamda, diam=diameter, scale_unit=galsim.arcsec,
                       obscuration=0.3, gsparams=gsparams)
    pixel = galsim.Pixel(pixel_scale, gsparams=gsparams)
    psf = galsim.Convolve(airy, pixel)
    given_psf = psf.drawImage(scale=pixel_scale/psf_oversample, method='no_pixel')
    # Draw at galaxy scale for shape measurement
    psf_galsample = psf.drawImage(scale=pixel_scale/gal_oversample, method='no_pixel')
    psfii = galsim.InterpolatedImage(given_psf, gsparams=gsparams, x_interpolant=interpolant)

    # Load galaxy catalog and select galaxies
    cc = galsim.COSMOSCatalog(dir='/disks/shear15/KiDS/ImSim/pipeline/data/COSMOS_25.2_training_sample/', use_real=False)
    sersicfit = cc.param_cat['sersicfit']
    hlr, sn, q, phi = [sersicfit[:, i] for i in (1, 2, 3, 7)]
    # Choose large galaxies, reasonable sersic n
    paramcatIndeces = np.where(np.logical_and(hlr*np.sqrt(q) > 2.5, sn >= 0.5))[0][:ngal]
    gals = cc.makeGalaxy(paramcatIndeces, chromatic=False, gsparams=gsparams)
    # All bulge+disk profiles are drawn as sersic. bulgeHandling saved in miscCode
    gals = [removeBulge(gal) for gal in gals]  # Change bulge+disk to just disk

    paramrec = cc.getParametricRecord(paramcatIndeces)
    corr_sersicfit = paramrec['sersicfit']
    corr_bulgefit = paramrec['bulgefit'][:, :8]
    corr_usebulgefit = paramrec['use_bulgefit']
    corr_usebulgefit = corr_usebulgefit.reshape(len(corr_usebulgefit), 1)
    corr_usebulgefit = np.repeat(corr_usebulgefit, 8, axis=1)
    use_sers = (1+corr_usebulgefit) % 2
    corr_fit_params = corr_sersicfit * use_sers + corr_bulgefit*corr_usebulgefit

    hlr, sn, q, phi = [corr_fit_params[:, i] for i in (1, 2, 3, 7)]
    ident = cc.getParametricRecord(paramcatIndeces)['IDENT']
    hlr = hlr * np.sqrt(q) * 0.03  # convert to circularized hlr in as

    # Option to circularize galaxies
    if circ:
        gals = [mcal.circularize(gal) for gal in gals]
        q = np.ones_like(sn)
    # Option to rotate galaxies to a particular orientation wrt coordinate axes
    if rotGals:
        gals = [mcal.rotGal(gal, intrinsicAngle, rotAngleRad)
                for gal, intrinsicAngle in zip(gals, phi)]
        phi = np.ones_like(phi) * rotAngleRad

    # Measure e1, e2
    e1arr, e2arr = measurement(gals, psf, psfii, psf_galsample, pixel_scale,
                               gal_oversample, shearList, gsparams, interpolant)
    res = [e1arr, e2arr, shearList, hlr, sn, q, phi, ident]  # , gals]

    if save:
        # Prepare to save data
        basedir = '/home/rosenberg/Documents/wl-bias-leaps-top/shear_bias_outputs/'
        basename = basedir + 'measErrs_diskonly'
        if circ:
            basename = basename + '_circ'
        if rotGals:
            basename = basename+'_rot'+str(rotAngleDeg)+'deg'
        name = basename + '_lambda%d' % (lamda)
        shearname = str((shearList[0].e1, shearList[0].e2))
        shearname = shearname.replace(', ', '-')[1:-1]
        name = name + '_shear%s' % (shearname)
        savename = name + '_%dgals_%s_%s.pkl' % (ngal, method, interpolant)

        # Save data
        fil = open(savename, 'wb')
        cPickle.dump(res, fil)
        fil.close()
        print 'Saved %s' % savename
    return res


# #parameters#

# gsparams = galsim.GSParams(kvalue_accuracy=1.e-5, maximum_fft_size=2048*10,
#                           maxk_threshold=9.e-4)

# pixel_scale = 0.1  # as/px
# lamda = 800  # nm # 550, 800, 920
# diameter = 1.2  # m
# psf_oversample = 5.
# gal_oversample = 2.
ngal = 10000
shearList = [galsim.shear.Shear(e1=0.0, e2=0)]
methodI = 0
methods = ['shearfirst', 'metacalshear']
measurements = [mcal.measure_shearfirst, mcal.measure_metacal]

method, measurement = methods[methodI], measurements[methodI]

# # options##
# circ = False
# bulgeHandling = 0  # Only sersic implemented
rotGals = True
rotAngleDeg = 0  # Degrees


if __name__ == '__main__':
    res = main(ngal, shearList, method, measurement, rotGals, rotAngleDeg)
