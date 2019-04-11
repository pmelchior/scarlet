import torch
from . import convolution


def build_detection_coadd(sed, bg_rms, observation, scene, thresh=1):
    B = observation.B
    images = observation.get_scene(scene)
    weights = torch.tensor([sed[b]/bg_rms[b]**2 for b in range(B)])
    jacobian = torch.tensor([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
    detect = torch.einsum('i,i...', weights, images) / jacobian

    # thresh is multiple above the rms of detect (weighted variance across bands)
    bg_cutoff = thresh * torch.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
    return detect, bg_cutoff


class Scene():
    # extent and characteristics of the modeled scence
    def __init__(self, shape, wcs=None):
        self._shape = tuple(shape)
        self.wcs = wcs

    @property
    def B(self):
        return self.shape[0]

    @property
    def Ny(self):
        return self.shape[1]

    @property
    def Nx(self):
        return self.shape[2]

    @property
    def shape(self):
        return self._shape

    def get_pixel(self, coord):
        if not isinstance(coord, torch.Tensor):
            coord = torch.tensor(coord)
        if self.wcs is not None:
            return self.wcs.radec2pix(coord).int()
        return coord.int()


class Observation(Scene):
    def __init__(self, images, psfs=None, weights=None, wcs=None, padding=3):
        super(Observation, self).__init__(images.shape, wcs=wcs)

        self.images = images
        self.psfs = psfs
        self.padding = padding

        if psfs is not None:
            ipad, ppad = convolution.get_common_padding(images, psfs, padding=padding)
            self.image_padding, self.psf_padding = ipad, ppad
            _psfs = torch.nn.functional.pad(psfs, self.psf_padding)
            self.psfs_fft = torch.rfft(_psfs, 2)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = 1

    def get_model(self, model, scene):
        if self.wcs is not None or scene.shape != self.shape:
            msg = "get_model is currently only supported when the observation frame matches the scene"
            raise NotImplementedError(msg)

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

    def get_loss(self, model, scene):
        if self.psfs is not None:
            model = self.get_model(model, scene)
        model *= self.weights
        return 0.5 * torch.nn.MSELoss(reduction='sum')(model, self.images*self.weights)

    def get_scene(self, scene):
        if self.wcs is not None or scene.shape != self.shape:
            msg = "get_scene is currently only supported when the observation frame matches the scene"
            raise NotImplementedError(msg)
        return self.images
