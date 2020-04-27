import numpy as np
from numpy.testing import assert_array_almost_equal
import scarlet
import pickle

def setup_scarlet(data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, channels, coverage='union'):
    '''Performs the initialisation steps for scarlet to run its resampling scheme
    Prameters
    ---------
    data_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    data_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    psf_hr: numpy array
        psf of the high resolution image
    psf_lr: numpy array
        psf of the low resolution image
    channels: tuple
        names of the channels
    Returns
    -------
    obs: array of observations
        array of scarlet.Observation objects initialised for resampling
    '''
    # Extract data
    im_hr = data_hr.array[None, :, :]
    im_lr = data_lr.array[None, :, :]
    # define two observation objects and match to frame
    obs_hr = scarlet.Observation(im_hr, wcs=data_hr.wcs, psfs=psf_hr, channels=[channels[1]])
    obs_lr = scarlet.Observation(im_lr, wcs=data_lr.wcs, psfs=psf_lr, channels=[channels[0]])
    # Keep the order of the observations consistent with the `channels` parameter
    # This implementation is a bit of a hack and will be refined in the future
    obs = [obs_lr, obs_hr]
    scarlet.Frame.from_observations(obs, obs_id=1, coverage=coverage)
    return obs

class TestLowResObservation(object):
    def test_surveys(self):
        # Load data for Euclid and Rubin Obs resolutions
        datas = np.load('../data/test_resampling/Multiresolution_tests.npz', 'r')

        # Tests for larger hr frame, smaller hr frame and same frames.
        # Each frame combination is tested for 'union' and 'intersection'
        for i in range(len(datas)-1):
            data_hr = datas[i]['data']
            psf_hr = datas[i]['PSF']
            wcs_hr = datas[i]['WCS']
            for j in range(len(datas[i + 1:])):
                data_lr = datas[j]['data']
                psf_lr = datas[j]['PSF']
                wcs_lr = datas[j]['WCS']

                obs_lr, obs_hr = setup_scarlet(data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, ['hr', 'lr'],
                                               coverage='union')
                interp_scar = obs_lr.render(data_hr.array[None, :, :])
                assert_array_almost_equal(interp_scar, data_lr)

                obs_lr, obs_hr = setup_scarlet(data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, ['hr', 'lr'],
                                               coverage='union')
                interp_scar = obs_lr.render(data_hr.array[None, :, :])
                assert_array_almost_equal(interp_scar, data_lr)

        def test_padded_frame(self):
            # Load data for Euclid and Rubin Obs resolutions
            datas = np.load('../data/test_resampling/Multiresolution_padded_tests.npz', 'r')

            # Tests for larger hr frame, smaller hr frame and same frames.
            # Each frame combination is tested for 'union' and 'intersection'
            for i in range(len(datas) - 1):
                data_hr = datas[i]['data']
                psf_hr = datas[i]['PSF']
                wcs_hr = datas[i]['WCS']
                for j in range(len(datas[i + 1:])):
                    data_lr = datas[j]['data']
                    psf_lr = datas[j]['PSF']
                    wcs_lr = datas[j]['WCS']

                    obs_lr, obs_hr = setup_scarlet(data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, ['hr', 'lr'],
                                                   coverage='union')
                    interp_scar = obs_lr.render(data_hr.array[None, :, :])
                    assert_array_almost_equal(interp_scar, data_lr)

                    obs_lr, obs_hr = setup_scarlet(data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, ['hr', 'lr'],
                                                   coverage='union')
                    interp_scar = obs_lr.render(data_hr.array[None, :, :])
                    assert_array_almost_equal(interp_scar, data_lr)