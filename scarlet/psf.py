import autograd.numpy as np


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


def generate_psf_image(func, shape, center=None, *args, **kwargs):
    """Generate a PSF image based on a function and shape
    """
    X = np.arange(shape[1])
    Y = np.arange(shape[0])
    X, Y = np.meshgrid(X, Y)
    coords = np.stack([Y, X])
    if center is None:
        center = (shape[0]-1) // 2, (shape[1]-1) // 2
    return func(coords, center[0], center[1], *args, **kwargs)


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
    return target_psf, all_params, params
