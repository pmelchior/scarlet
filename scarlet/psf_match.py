import numpy as np
import torch

from .config import Config
from . import constraint as sc
from .component import Component
from .source import Source
from .blend import Blend
from .observation import Observation


def moffat(coords, y0, x0, amplitude, alpha, beta=1.5):
    """Moffat Function

    Symmetric 2D Moffat function:

    .. math::

        A (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}
    """
    Y, X = coords
    return (amplitude*(1+((X-x0)**2+(Y-y0)**2)/alpha**2)**-beta)


def gaussian(coords, y0, x0, amplitude, sigma):
    """Circular Gaussian Function
    """
    Y, X = coords
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

    psfs = psfs.detach().numpy()
    X = np.arange(psfs.shape[2])
    Y = np.arange(psfs.shape[1])
    X, Y = np.meshgrid(X, Y)
    coords = np.stack([Y, X])
    y0, x0 = (psfs.shape[1]-1) // 2, (psfs.shape[2]-1) // 2
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
            params.append(np.mean(all_params[:, 2]))  # amplitude
            params.append(np.min(all_params[:, 3])*0.6)  # alpha
            params.append(np.max(all_params[:, 4])*1.4)  # beta
        elif func == gaussian:
            params.append(np.mean(all_params[:, 2]))  # amplitude
            params.append(np.min(all_params[:, 3])*0.6)  # sigma
        elif func == double_gaussian:
            params.append(np.mean(all_params[:, 2]))  # amplitude 1
            params.append(np.min(all_params[:, 3])*0.6)  # sigma 1
            params.append(np.mean(all_params[:, 4]))  # amplitude 2
            params.append(np.min(all_params[:, 5])*0.6)  # sigma 2
    else:
        params = extract_values(all_params)
    target_psf = func(coords, y0, x0, *params)

    # normalize the target PSF
    target_psf = target_psf/target_psf.sum()
    return torch.tensor(target_psf.astype(np.float32)), all_params, params


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
    config = Config(refine_skip=100)
    center = np.array([psfs[0].shape[0] // 2, psfs[0].shape[1] // 2])
    if constraints is None:
        constraints = [sc.MinimalConstraint()]
        if l0_thresh > 0:
            constraints.append(sc.L0Constraint(l0_thresh))
    if cutoff is None:
        cutoff = 0
    sources = [
        PSFDiffKernel(center, psfs, cutoff, b, constraints=constraints, config=config)
        for b in range(len(psfs))
    ]
    target_psf = torch.stack([target_psf for n in range(len(psfs))])
    observation = Observation(images=psfs, psfs=target_psf, bg_rms=[cutoff]*len(psfs))
    psf_blend = Blend(sources, observation, config)
    psf_blend.fit(max_iter, e_rel=1e-3, padding=3)
    diff_kernels = torch.stack([kernel.morph for kernel in psf_blend.components])
    return diff_kernels, psf_blend


class PSFDiffKernel(Source):
    """Create a model of the PSF in a single band

    Passing a `PSFDiffKernel` for each band as a list of sources to
    the :class:`scarlet.blend.Blend` class can be used to model
    the difference kernel in each band, which gives more accurate
    results when performing PSF deconvolution.
    """
    def __init__(self, center, psfs, cutoff, band, constraints=None, config=None):
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
        config: :class:`scarlet.config.Config` instance, default=`None`
            Special configuration to overwrite default optimization parameters

        see :class:`scarlet.source.Source` for other parameters.
        """

        # set sed and morph to that of `band`
        B, Ny, Nx = psfs.shape
        sed = torch.zeros(B)
        sed[band] = 1
        morph = psfs[band]

        component = Component(sed, morph, center=center, constraints=constraints,
                              fix_sed=True, fix_morph=False)
        super(PSFDiffKernel, self).__init__(component)
