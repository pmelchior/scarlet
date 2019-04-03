import torch
from . import convolution


class Observation(object):
    def __init__(self, images, psfs=None, masks=None, weights=None, bg_rms=None, padding=3):
        self.B, self.Ny, self.Nx = images.shape
        self.images = images
        self.psfs = psfs
        self.masks = masks
        self.weights = weights
        self.padding = padding

        if bg_rms is None:
            self._bg_rms = torch.zeros(self.B, dtype=images.dtype)
        else:
            assert len(bg_rms) == self.B
            if not isinstance(bg_rms, torch.Tensor):
                self._bg_rms = torch.tensor(bg_rms, dtype=images.dtype)

        if psfs is not None:
            ipad, ppad = convolution.get_common_padding(images, psfs, padding=padding)
            self.image_padding, self.psf_padding = ipad, ppad
            _psfs = torch.nn.functional.pad(psfs, self.psf_padding)
            self.psfs_fft = torch.rfft(_psfs, 2)

    def convolve_model(self, model):
        def _convolve_band(model, psf):
            _model = torch.nn.functional.pad(model, self.image_padding)
            Image = torch.rfft(_model, 2)
            Convolved = convolution.complex_mul(Image, psf)
            convolved = torch.irfft(Convolved, 2, signal_sizes=_model.shape)
            result = convolution.ifftshift(convolved)
            bottom, top, left, right = self.image_padding
            result = result[bottom:-top, left:-right]
            return result
        return torch.stack([_convolve_band(model[b], self.psfs_fft[b]) for b in range(self.B)])

    def gradient(self, model):
        if self.psfs is not None:
            model = self.convolve_model(model)
        torch.nn.MSELoss(reduction='sum')(model, self.images).backward()

    def _set_weights(self, weights):
        """Set the weights and pixel covariance matrix `_Sigma_1`.

        Parameters
        ----------
        weights: array-like
            Array (Band, Height, Width) of weights for each image, in each band

        Returns
        -------
        None, but sets `self._Sigma_1` and `self._weights`.
        """
        if weights is None:
            self._weights = [1, 1]
        else:
            self._weights = [None] * 2

            # for S update: normalize the per-pixel variation
            # i.e. in every pixel: utilize the bands with large weights
            # CAVEAT: need to filter out pixels that are saturated in every band
            norm_pixel = weights.median(dim=0)
            mask = norm_pixel > 0
            self._weights[1] = weights.clone()
            self._weights[1][:, mask] /= norm_pixel[mask]

            # reverse is true for A update: for each band, use the pixels that
            # have the largest weights
            norm_band = weights.reshape(self.B, -1).median(dim=1)
            # CAVEAT: some regions may have one band missing completely
            mask = norm_band > 0
            self._weights[0] = weights.clone()
            self._weights[0][mask] /= norm_band[mask, None, None]
            # CAVEAT: mask all pixels in which at least one band has W=0
            # these are likely saturated and their colors have large weights
            # but are incorrect due to missing bands
            mask = ~torch.all(weights > 0, axis=0)
            # and mask all bands for that pixel:
            # when estimating A do not use (partially) saturated pixels
            self._weights[0][:, mask] = 0
