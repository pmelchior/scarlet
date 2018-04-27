import numpy as np

from .config import Config
from . import constraints as sc
from .source import Source, ExtendedSource
from .blend import Blend

def moffat(coords, y0, x0, amplitude, alpha, beta=1.5):
    """Moffat Function

    Symmetric 2D Moffat function:

    .. math::
        
        A (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}
    """
    Y,X = coords
    return (amplitude*(1+((X-x0)**2+(Y-y0)**2)/alpha**2)**-beta)

def gaussian(coords, y0, x0, amplitude, sigma):
    """Circular Gaussian Function
    """
    Y,X = coords
    return (amplitude*np.exp(-((X-x0)**2+(Y-y0)**2)/(2*sigma**2)))

def double_gaussian(coords, y0, x0, A1, sigma1, A2, sigma2):
    """Sum of two Gaussian Functions
    """
    return gaussian(coords, y0, x0, A1, sigma1) + gaussian(coords, y0, x0, A2, sigma2)

def fit_target_psf(psfs, func, init_values=None, extract_values=None):
    """Build a target PSF from a collection of psfs

    Parameters
    ----------
    psfs: array
        Array of 2D psf images.
    func: function
        Function to fit each `psfs`.
        If `func` is not `moffat`, `gaussian`, or `double_gaussian`,
        then `init_values` and `extract_values` must be specified.
    init_values: list
        Initial values of `func` to use in the fit.
        If `init_values` is `None` then the following default values are used:
        * `moffat`: `amplitude=1`, `alpha=3`, `beta=1.5`
        * `gaussian`: `amplitude=1`, `sigma=3`
        * `double_gaussian`: `A1=1`, `sigma1=3`, `A2=.5`, `sigma2=6`
        * custom function: `ValueError`
    extract_values: func
        Function to use to extract the parameters for the `target_psf`.
        `extract_values` should take a single argument `all_params`, an array
        that contains the list of parameters for each PSF.
        If `extract_values` is `None` then the following schemes are used:
        * `moffat`: `alpha=min(alpha)*0.6`, `beta=max(beta)*1.4`
        * `gaussian`: `sigma=min(sigma)*0.6`
        * `double_gaussian`: `sigma1=min(sigma1)*0.6`, `sigma2=min(sigma2)*0.6`
        * custom function: `ValueError`

    Results
    -------
    target_psf: array
        Smaller target PSF to use for de-convolving `psfs`.
    """
    from scipy.optimize import curve_fit

    X = np.arange(psfs.shape[2])
    Y = np.arange(psfs.shape[1])
    X,Y = np.meshgrid(X,Y)
    coords = np.array([Y,X])
    y0, x0 = psfs.shape[1]//2, psfs.shape[2]//2
    init_params = [y0, x0]
    if init_values is None:
        if func == moffat:
            init_params += [1., 3., 1.5]
        elif func == gaussian:
            init_params += [1., 3.]
        elif func == double_gaussian:
            init_params += [1., 3., .5, 6.]
        else:
            raise ValueError("Custom functions require `init_values`")
    all_params = []
    init_params = tuple(init_params)

    def reshape_func(*args):
        """Flatten the function output to work with `scipy.curve_fit`
        """
        return func(*args).reshape(-1)

    # Fit each PSF with the specified function and store the results
    for n, psf in enumerate(psfs):
        params, cov = curve_fit(reshape_func, coords, psf.reshape(-1), p0=init_params)
        all_params.append(params)
    all_params = np.array(all_params)

    # Use the appropriate scheme to choose the ideal target PSF
    # based on the best-fit parameters for each PSF
    if extract_values is None:
        params = []
        if func == moffat:
            params.append(np.mean(all_params[:,2])) # amplitude
            params.append(np.min(all_params[:,3])*0.6) # alpha
            params.append(np.max(all_params[:,4])*1.4) # beta
        elif func == gaussian:
            params.append(np.mean(all_params[:,2])) # amplitude
            params.append(np.min(all_params[:,3])*0.6) # sigma
        elif func == double_gaussian:
            params.append(np.mean(all_params[:,2])) # amplitude 1
            params.append(np.min(all_params[:,3])*0.6) # sigma 1
            params.append(np.mean(all_params[:,4])) # amplitude 2
            params.append(np.min(all_params[:,5])*0.6) # sigma 2
    else:
        params = extract_values(all_params)
    target_psf = func(coords, y0, x0, *params)

    # normalize the target PSF
    target_psf = target_psf/np.sum(target_psf)
    return target_psf

def build_diff_kernels(psfs, target_psf, max_iter=100, e_rel=1e-3, constraints=None, cutoff=None,
                       l0_thresh=1e-4):
    """Build the difference kernel to match a list of `psfs` to a `target_psf`

    This convenience function runs the `Blend` class on a collection of
    `PSFDiffKernel` objects as sources to match `psfs` to `target_psf`.

    Parameters
    ----------
    psfs: array
        Array of 2D psf images to match to the `target_psf`.
    target_psf: array
        Target PSF to use for de-convolving `psfs`.
    max_iter: int
        Maximum number of iterations used to create the difference kernels
    e_rel: float
        Relative error to use when matching the PSFs
    constraints: `Constraint` or `ConstraintList`
        Constraints used to match the PSFs.
        If `constraints` is `None` then `SimpleConstraint` and `L0Constraint`
        are used.
    `cutoff`: floats
        Minimum non-zero value of the difference kernel for each PSF.
        If `cutoff` is `None`, then each PSF has no minimum value set,
        which is the recommended value.

    Returns
    -------
    diff_kernels: array
        Array of 2D difference kernels for each PSF in `psfs`
    `psf_blend`: `Blend`
        `Blend` object that contains the result of matching
        `psfs` to `target_psf`, where `diff_kernels` is an array
        of the `Source.morph` for each source in `psf_blend`.
    """
    config = Config(refine_skip=100, source_sizes=np.array([np.max(psfs.shape[1:])]))
    center = np.array([psfs[0].shape[0] // 2, psfs[0].shape[1] //2], dtype=psfs.dtype)
    if constraints is None:
        constraints = sc.SimpleConstraint() & sc.L0Constraint(l0_thresh)
    if cutoff is None:
        cutoff = 0
    sources = [
        PSFDiffKernel(center, psfs, cutoff, target_psf, b, constraints=constraints.copy(),
                      monotonic=False, config=config) for b in range(len(psfs))
    ]
    psf_blend = Blend(sources, psfs, bg_rms=[cutoff]*len(psfs), config=config)
    psf_blend.fit(100, e_rel=1e-3)
    diff_kernels = np.array([kernel.morph for kernel in psf_blend.sources]).reshape(psfs.shape)
    return diff_kernels, psf_blend

class PSFDiffKernel(ExtendedSource):
    """Create a model of the PSF in a single band

    Passing a `PSFDiffKernel` for each band as a list of sources to
    the :class:`scarlet.blend.Blend` class can be used to model
    the difference kernel in each band, which gives more accurate
    results when performing PSF deconvolution.
    """
    def __init__(self, center, psf, cutoff, target_psf, band,
                 constraints=None, monotonic=True, config=None, fix_frame=True, shift_center=0.0):
        """Initialize the difference kernel in a single band

        See :class:`~scarlet.source.Source` for parameter descriptions not listed below.

        Parameters
        ----------
        cutoff: array-like
            Minimum values of a psf in any given band, used to contrain
            the initial size of the PSF.
        target_psf: array-like
            The target PSF for all of the bands
        band: int
            Each `PSFDiffKernel` has a fixed SED with only a single
            non-zero band, where `band` is the index in the multi-band
            data corresponding to this `PSFDiffKernel`.
        monotonic: bool
            Whether or not to make the initial difference kernel monotonic
        config: :class:`scarlet.config.Config` instance, default=`None`
            Special configuration to overwrite default optimization parameters

        see :class:`scarelt.source.Source` for other parameters.
        """
        self.center = center
        sed, morph = self._make_initial(psf, cutoff, target_psf, band, monotonic=monotonic, config=config)

        if constraints is None:
            constraints = (sc.SimpleConstraint() &
                           sc.DirectMonotonicityConstraint(use_nearest=False))

        Source.__init__(self, sed=sed, morph_image=morph, center=center, constraints=constraints,
                        fix_sed=True, fix_morph=False, fix_frame=fix_frame,
                        shift_center=shift_center, psf=target_psf)

    def _make_initial(self, psf, cutoff, target_psf, band, monotonic=True, config=None):
        """Initialize the source that is symmetric and monotonic
        """
        # Use a default configuration if config is not specified
        if config is None:
            config = Config(source_sizes=np.array([np.max(psf.shape[1:])]))
        # every source as large as the entire image, but shifted to its centroid
        B, Ny, Nx = psf.shape
        self._set_frame(self.center, (Ny,Nx))

        # determine initial SED from peak position
        sed = np.zeros((B,))
        sed[band] = 1

        # copy morph from detect cutout, make non-negative
        source_slice = self.get_slice_for(psf.shape)
        morph = np.zeros((self.Ny, self.Nx))
        morph[source_slice[1:]] = psf[band, self.bb[1], self.bb[2]].copy()

        morph = self._init_morph(morph, source_slice, cutoff,
                                 symmetric=False, monotonic=monotonic, config=config)

        return sed.reshape((1,B)), morph.reshape((1, morph.shape[0], morph.shape[1]))
