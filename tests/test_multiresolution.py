import numpy as np
import scarlet
import os


def SDR(X_true, X):
    """Source distortion ratio between an expected value and its estimate. The higher the SDR the better X_true and X agree"""
    return 10 * np.log10(np.sum(X_true ** 2) ** 0.5 / np.sum((X_true - X) ** 2) ** 0.5)


def setup_scarlet(
    data_hr, wcs_hr, data_lr, wcs_lr, psf_hr, psf_lr, channels, coverage="union"
):
    """Performs the initialisation steps for scarlet to run its resampling scheme
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
    """
    # Extract data
    im_hr = data_hr[None, :, :]
    im_lr = data_lr[None, :, :]
    # define two observation objects and match to frame
    obs_hr = scarlet.Observation(
        im_hr, wcs=wcs_hr, psf=scarlet.ImagePSF(psf_hr), channels=[channels[1]]
    )
    obs_lr = scarlet.Observation(
        im_lr, wcs=wcs_lr, psf=scarlet.ImagePSF(psf_lr), channels=[channels[0]]
    )
    # Keep the order of the observations consistent with the `channels` parameter
    # This implementation is a bit of a hack and will be refined in the future
    obs = [obs_lr, obs_hr]
    scarlet.Frame.from_observations(obs, obs_id=1, coverage=coverage)
    return obs


class TestLowResObservation(object):
    def test_surveys(self):
        # Load data for Euclid and Rubin Obs resolutions
        datas = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../data/test_resampling/Multiresolution_tests.npz",
            ),
            allow_pickle=True,
        )
        images = datas["images"]
        psfs = datas["psf"]
        wcss = datas["wcs"]

        # Tests for larger hr frame, smaller hr frame and same frames.
        # Each frame combination is tested for 'union' and 'intersection'
        for i in range(len(images)):
            data_hr = images[i]
            psf_hr = psfs[i]
            wcs_hr = wcss[i]
            # Reconstructs the array shape because in storrage
            wcs_hr.array_shape = wcs_hr.wcs.crpix * 2

            for j in np.arange(i + 1, len(images)):
                data_lr = images[j]
                psf_lr = psfs[j]
                wcs_lr = wcss[j]
                wcs_lr.array_shape = wcs_lr.wcs.crpix * 2

                obs_lr, obs_hr = setup_scarlet(
                    data_hr,
                    wcs_hr,
                    data_lr,
                    wcs_lr,
                    psf_hr,
                    psf_lr,
                    ["hr", "lr"],
                    coverage="union",
                )
                interp_scar = obs_lr.render(data_hr[None, :, :])
                assert SDR(interp_scar, data_lr) > 10

                obs_lr, obs_hr = setup_scarlet(
                    data_hr,
                    wcs_hr,
                    data_lr,
                    wcs_lr,
                    psf_hr,
                    psf_lr,
                    ["hr", "lr"],
                    coverage="intersection",
                )
                interp_scar = obs_lr.render(data_hr[None, :, :])
                assert SDR(interp_scar, data_lr) > 10

    def test_padded_frame(self):
        # Load data for Euclid and Rubin Obs resolutions
        datas = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../data/test_resampling/Multiresolution_tests.npz",
            ),
            allow_pickle=True,
        )
        images = datas["images"]
        psfs = datas["psf"]
        wcss = datas["wcs"]
        # Tests for larger hr frame, smaller hr frame and same frames.
        # Each frame combination is tested for 'union' and 'intersection'
        data_hr = images[0]
        psf_hr = psfs[0]
        wcs_hr = wcss[0]
        wcs_hr.array_shape = wcs_hr.wcs.crpix * 2
        for j in range(len(images) - 1):
            data_lr = images[j + 1]
            psf_lr = psfs[j + 1]
            wcs_lr = wcss[j + 1]
            wcs_lr.array_shape = wcs_lr.wcs.crpix * 2
            obs_lr, obs_hr = setup_scarlet(
                data_hr,
                wcs_hr,
                data_lr,
                wcs_lr,
                psf_hr,
                psf_lr,
                ["hr", "lr"],
                coverage="union",
            )
            interp_scar = obs_lr.render(data_hr[None, :, :])
            assert SDR(interp_scar, data_lr) > 10
            obs_lr, obs_hr = setup_scarlet(
                data_hr,
                wcs_hr,
                data_lr,
                wcs_lr,
                psf_hr,
                psf_lr,
                ["hr", "lr"],
                coverage="intersection",
            )
            interp_scar = obs_lr.render(data_hr[None, :, :])
            assert SDR(interp_scar, data_lr) > 10
