import numpy as np
from astropy.visualization.lupton_rgb import LinearMapping, AsinhMapping
import matplotlib.pyplot as plt
from .component import ComponentTree

def get_default_filter_weight(bands, channels=3):
    filter_weights = np.zeros((channels, bands))
    if bands == 1:
        filter_weights[0, 0] = filter_weights[1, 0] = filter_weights[2, 0] = 1
    elif bands == 2:
        filter_weights[0, 1] = 0.667
        filter_weights[1, 1] = 0.333
        filter_weights[1, 0] = 0.333
        filter_weights[2, 0] = 0.667
        filter_weights /= 0.667
    elif bands == 3:
        filter_weights[0, 2] = 1
        filter_weights[1, 1] = 1
        filter_weights[2, 0] = 1
    elif bands == 4:
        filter_weights[0, 3] = 1
        filter_weights[0, 2] = 0.333
        filter_weights[1, 2] = 0.667
        filter_weights[1, 1] = 0.667
        filter_weights[2, 1] = 0.333
        filter_weights[2, 0] = 1
        filter_weights /= 1.333
    elif bands == 5:
        filter_weights[0, 4] = 1
        filter_weights[0, 3] = 0.667
        filter_weights[1, 3] = 0.333
        filter_weights[1, 2] = 1
        filter_weights[1, 1] = 0.333
        filter_weights[2, 1] = 0.667
        filter_weights[2, 0] = 1
        filter_weights /= 1.667
    elif bands == 6:
        filter_weights[0, 5] = 1
        filter_weights[0, 4] = 0.667
        filter_weights[0, 3] = 0.333
        filter_weights[1, 4] = 0.333
        filter_weights[1, 3] = 0.667
        filter_weights[1, 2] = 0.667
        filter_weights[1, 1] = 0.333
        filter_weights[2, 2] = 0.333
        filter_weights[2, 1] = 0.667
        filter_weights[2, 0] = 1
        filter_weights /= 2
    else:
        raise NotImplementedError("No default filter weights have been implemented for more than 6 bands")
    return filter_weights


class LinearPercentileNorm(LinearMapping):
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhPercentileNorm(AsinhMapping):
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        # solution for beta assumes flat spectrum at vmax
        stretch = vmax - vmin
        beta = stretch / np.sinh(1)
        super().__init__(minimum=vmin, stretch=stretch, Q=beta)


def img_to_channel(img, filter_weights=None, fill_value=0):
    """Convert multi-band image cube into 3 RGB channels

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (bands, height, width).
    filter_weights: array_like
        Linear mapping with dimensions (channels, bands)
    fill_value: float, default=`0`
        Value to use for any masked pixels.

    Returns
    -------
    RGB: numpy array with dtype float
    """
    # expand single img into cube
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    B = len(img_)
    C = 3  # RGB

    # filterWeights: channel x band
    if filter_weights is None:
        filter_weights = get_default_filter_weight(B, C)
    else:
        assert filter_weights.shape == (3, len(img))

    # map bands onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(filter_weights, img_.reshape(B, -1)).reshape(3, ny, nx)

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(img, filter_weights=None, fill_value=0, norm=None):
    """Convert images to normalized RGB.

    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (bands, height, width).
    filter_weights: array_like
        Linear mapping with dimensions (channels, bands)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    norm: `scarlet.display.Norm`, default `None`
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.

    Returns
    -------
    rgb: numpy array with dimensions (3, height, width) and dtype uint8
    """
    RGB = img_to_channel(img, filter_weights=filter_weights)
    if norm is None:
        norm = LinearMapping(image=RGB)
    rgb = norm.make_rgb_image(*RGB)
    return rgb

def show_scene(sources, observation=None, norm=None, filter_weights=None, show_observed=False, show_rendered=False, show_residual=False, label_sources=True, figsize=None):
    if show_observed or show_rendered or show_residual:
        assert observation is not None, "Provide matched observation to show observed frame"

    panels = 1 + sum((show_observed, show_rendered, show_residual))
    if figsize is None:
        figsize = (3*panels, 3*len(list(sources)))
    fig, ax = plt.subplots(1, panels, figsize=figsize)
    if not hasattr(ax, '__iter__'):
        ax = (ax,)

    panel = 0
    tree = ComponentTree(sources)
    model = tree.get_model()
    ax[panel].imshow(img_to_rgb(model, norm=norm, filter_weights=filter_weights))
    ax[panel].set_title("Model")

    if show_rendered or show_residual:
        model = observation.render(model)

    if show_rendered:
        panel += 1
        ax[panel].imshow(img_to_rgb(model, norm=norm, filter_weights=filter_weights))
        ax[panel].set_title("Model Rendered")

    if show_observed:
        panel += 1
        ax[panel].imshow(img_to_rgb(observation.images, norm=norm, filter_weights=filter_weights))
        ax[panel].set_title("Observation")

    if show_residual:
        panel += 1
        residual = observation.images - model
        norm_ = LinearPercentileNorm(residual)
        ax[panel].imshow(img_to_rgb(residual, norm=norm_, filter_weights=filter_weights))
        ax[panel].set_title("Residual")

    if label_sources:
        for k, src in enumerate(sources):
            if hasattr(src, 'center'):
                center = np.array(src.center)
                center_ = center - np.array(src.frame.origin[1:]) # observed coordinates
            ax[0].text(*center[::-1], k, color='w')
            for panel in range(1, panels):
                ax[panel].text(*center_[::-1], k, color='w')

    fig.tight_layout()
    plt.close();
    return fig

def show_sources(sources, observation=None, norm=None, filter_weights=None, show_observed=False, show_rendered=False, show_sed=True, figsize=None):

    if show_observed or show_rendered:
        assert observation is not None, "Provide matched observation to show observed frame"

    panels = 1 + sum((show_observed, show_rendered, show_sed))
    if figsize is None:
        figsize = (3*panels, 3*len(list(sources)))
    fig, ax = plt.subplots(len(list(sources)), panels, figsize=figsize)
    for k,src in enumerate(sources):

        if hasattr(src, 'center'):
            center = np.array(src.center)
            # center in src bbox coordinates
            if src.bbox is not None:
                center_ = center - np.array(src.bbox.origin[1:])
            else:
                center_ = center
            # center in observed coordinates
            center__ = center - np.array(src.frame.origin[1:])
        else:
            center = None

        panel = 0
        frame_ = src.frame
        src.set_frame(src.bbox)
        if isinstance(src, ComponentTree):
            model = 0
            seds = []
            for component in src:
                model_ = component.get_model()
                seds.append(model_.sum(axis=(1,2)))
                model += model_
        else:
            model = src.get_model()
            seds = [model.sum(axis=(1,2))]
        src.set_frame(frame_)
        ax[k][panel].imshow(img_to_rgb(model, norm=norm, filter_weights=filter_weights))
        ax[k][panel].set_title("Model Source {}".format(k))
        if center is not None:
            ax[k][panel].plot(*center_[::-1], "wx", mew=1, ms=10)

        if show_rendered:
            panel += 1
            model = src.get_model()
            model = observation.render(model)
            ax[k][panel].imshow(img_to_rgb(model, norm=norm, filter_weights=filter_weights))
            ax[k][panel].set_xlim(src.bbox.left, src.bbox.right)
            ax[k][panel].set_ylim(src.bbox.bottom, src.bbox.top)
            ax[k][panel].set_title("Model Source {} Rendered".format(k))
            if center is not None:
                ax[k][panel].plot(*center__[::-1], "wx", mew=1, ms=10)

        if show_observed:
            panel += 1
            ax[k][panel].imshow(img_to_rgb(observation.images, norm=norm, filter_weights=filter_weights))
            ax[k][panel].set_xlim(src.bbox.left, src.bbox.right)
            ax[k][panel].set_ylim(src.bbox.bottom, src.bbox.top)
            ax[k][panel].set_title("Observation".format(k))
            if center is not None:
                ax[k][panel].plot(*center__[::-1], "wx", mew=1, ms=10)

        if show_sed:
            panel += 1
            for sed in seds:
                ax[k][panel].plot(sed)
            ax[k][panel].set_xticks(range(len(sed)))
            if hasattr(src.frame, 'channels'):
                ax[k][panel].set_xticklabels(src.frame.channels)
            ax[k][panel].set_title("SED")
            ax[k][panel].set_xlabel("Channel")
            ax[k][panel].set_ylabel("Intensity")

    fig.tight_layout()
    plt.close();
    return fig
