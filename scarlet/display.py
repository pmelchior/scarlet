import numpy as np
from astropy.visualization.lupton_rgb import LinearMapping, AsinhMapping
import matplotlib.pyplot as plt
from .component import ComponentTree
from .observation import convolve


def channels_to_rgb(channels):
    """Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the the channels are ordered in wavelength
    direction, starting with the shortest wavelength. The mapping seeks to produce
    a relatively even weights for across all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.
    Parameters
    ----------
    channels: int in range(0,7)
        Number of channels
    Returns
    -------
    array (3, channels) to map onto RGB
    """
    assert channels in range(
        0, 7
    ), "No mapping has been implemented for more than {} channels".format(channels)

    channel_map = np.zeros((3, channels))
    if channels == 1:
        channel_map[0, 0] = channel_map[1, 0] = channel_map[2, 0] = 1
    elif channels == 2:
        channel_map[0, 1] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[1, 0] = 0.333
        channel_map[2, 0] = 0.667
        channel_map /= 0.667
    elif channels == 3:
        channel_map[0, 2] = 1
        channel_map[1, 1] = 1
        channel_map[2, 0] = 1
    elif channels == 4:
        channel_map[0, 3] = 1
        channel_map[0, 2] = 0.333
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.667
        channel_map[2, 1] = 0.333
        channel_map[2, 0] = 1
        channel_map /= 1.333
    elif channels == 5:
        channel_map[0, 4] = 1
        channel_map[0, 3] = 0.667
        channel_map[1, 3] = 0.333
        channel_map[1, 2] = 1
        channel_map[1, 1] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 1.667
    elif channels == 6:
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    return channel_map


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


def img_to_3channel(img, channel_map=None, fill_value=0):
    """Convert multi-band image cube into 3 RGB channels
    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
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
    C = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(C)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(channel_map, img_.reshape(C, -1)).reshape(3, ny, nx)

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(img, channel_map=None, fill_value=0, norm=None, mask=None):
    """Convert images to normalized RGB.
    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.
    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    norm: `scarlet.display.Norm`, default `None`
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.
    mask: array_like
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.
    Returns
    -------
    rgb: numpy array with dimensions (3, height, width) and dtype uint8
    """
    RGB = img_to_3channel(img, channel_map=channel_map)
    if norm is None:
        norm = LinearMapping(image=RGB)
    rgb = norm.make_rgb_image(*RGB)
    if mask is not None:
        rgb = np.dstack([rgb, ~mask * 255])
    return rgb


def show_scene(
    sources,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_residual=False,
    label_sources=True,
    figsize=None,
    linear=True,
):
    """Plot all sources to recreate the scence.
    The functions provides a fast way of evaluating the quality of the entire model,
    i.e. the combination of all scences that seek to fit the observation.
    Parameters
    ----------
    sources: list of source models
    observation: `~scarlet.Observation`
    norm: norm to compress image intensity to the range [0,255]
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    show_model: bool
        Whether the model is shown in the model frame
    show_observed: bool
        Whether the observation is shown
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown
    show_residual: bool
        Whether the residuals between rendered model and observation is shown
    label_sources: bool
        Whether each source is labeled with its numerical index in the source list
    figsize: matplotlib figsize argument
    linear: bool
        Whether or not to display the scene in a single line (`True`) or
        on multiple lines (`False`).
    Returns
    -------
    matplotlib figure
    """
    if show_observed or show_rendered or show_residual:
        assert (
            observation is not None
        ), "Provide matched observation to show observed frame"

    panels = sum((show_model, show_observed, show_rendered, show_residual))
    if linear:
        if figsize is None:
            figsize = (3 * panels, 3 * len(list(sources)))
        fig, ax = plt.subplots(1, panels, figsize=figsize)
    else:
        columns = int(np.ceil(panels/2))
        if figsize is None:
            figsize = (7*columns, 4*columns)
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(2, columns, n+1) for n in range(panels)]
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    # Mask any pixels with zero weight in all bands
    if observation is not None:
        mask = np.sum(observation.weights, axis=0) == 0
        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None

    panel = 0
    if hasattr(sources, "components"):
        # The list of sources is already a tree, so just use it
        tree = sources
    else:
        tree = ComponentTree(sources)
    if observation is None:
        model = tree.get_model()
        extent = get_extent(tree.bbox)
    else:
        model = tree.get_model(frame=observation.frame)
        extent = get_extent(observation.frame)

    if show_model:
        ax[panel].imshow(img_to_rgb(model, norm=norm, channel_map=channel_map), extent=extent, origin="lower")
        ax[panel].set_title("Model")
        ax[panel].set_xlim(extent[0], extent[1])
        ax[panel].set_ylim(extent[2], extent[3])
        panel += 1

    if show_rendered or show_residual:
        model = observation.render(model)

    if show_rendered:
        ax[panel].imshow(
            img_to_rgb(model, norm=norm, channel_map=channel_map, mask=mask), extent=extent, origin="lower"
        )
        ax[panel].set_title("Model Rendered")
        ax[panel].set_xlim(extent[0], extent[1])
        ax[panel].set_ylim(extent[2], extent[3])
        panel += 1

    if show_observed:
        ax[panel].imshow(
            img_to_rgb(
                observation.images, norm=norm, channel_map=channel_map, mask=mask
            ), extent=extent, origin="lower"
        )
        ax[panel].set_title("Observation")
        ax[panel].set_xlim(extent[0], extent[1])
        ax[panel].set_ylim(extent[2], extent[3])
        panel += 1

    if show_residual:
        residual = observation.images - model
        norm_ = LinearPercentileNorm(residual)
        ax[panel].imshow(
            img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent, origin="lower"
        )
        ax[panel].set_title("Residual")
        ax[panel].set_xlim(extent[0], extent[1])
        ax[panel].set_ylim(extent[2], extent[3])
        panel += 1

    if label_sources:
        for k, src in enumerate(sources):
            if hasattr(src, "center"):
                center = np.array(src.center)
                center_ = center - np.array(
                    src.model_frame.origin[1:]
                )  # observed coordinates
            ax[0].text(*center[::-1], k, color="w")
            for panel in range(1, panels):
                ax[panel].text(*center_[::-1], k, color="w")

    fig.tight_layout()
    return fig


def get_extent(bbox):
    extent = np.array([bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]], dtype=float) - 0.5
    return extent


def show_sources(
    sources,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_sed=True,
    figsize=None,
    model_mask=None,
    mark_centers=True,
):
    """Plot each source individually.
    The functions provides an more detailed inspection of every source in the list.
    Parameters
    ----------
    sources: list of source models
    observation: `~scarlet.Observation`
    norm: norm to compress image intensity to the range [0,255]
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    show_model: bool
        Whether the model is shown in the model frame
    show_observed: bool
        Whether the observation is shown
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown
    show_sed: bool
        Whether source SED is shown.
        For multi-component sources, SEDs are shown separately.
    figsize: matplotlib figsize argument
    model_mask: array
        Mask used to hide pixels in the sparese model only.
        This can be used to understand how the underlying model is
        behaving, but defaults to `None`, which does not mask the
        model at all.
    mark_centers: bool
        Whether or not to mark the centers of the soures
        with their source number.
    Returns
    -------
    matplotlib figure
    """
    if show_observed or show_rendered:
        assert (
            observation is not None
        ), "Provide matched observation to show observed frame"

    panels = sum((show_model, show_observed, show_rendered, show_sed))
    if figsize is None:
        figsize = (3 * panels, 3 * len(list(sources)))
    fig, ax = plt.subplots(len(list(sources)), panels, figsize=figsize)
    for k, src in enumerate(sources):
        if hasattr(src, "center"):
            center = np.array(src.center)[::-1]
        else:
            center = None

        panel = 0

        model = src.get_model()
        if isinstance(src, ComponentTree):
            seds = []
            for component in src:
                model_ = component.get_model()
                seds.append(model_.sum(axis=(1, 2)))
        else:
            seds = [model.sum(axis=(1, 2))]

        extent = get_extent(src.bbox)
        rendered_box = src.bbox
        if show_model:
            # Show the unrendered model in it's bbox
            ax[k][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent, origin="lower")
            ax[k][panel].set_title("Model Source {}".format(k))
            ax[k][panel].set_xlim(extent[0], extent[1])
            ax[k][panel].set_ylim(extent[2], extent[3])
            if center is not None and mark_centers:
                ax[k][panel].plot(*center, "wx", mew=1, ms=10)
            panel += 1

        if show_rendered:
            # Center and show the rendered model
            model = src.model_to_frame(frame=rendered_box)
            model = observation.convolve(model)
            ax[k][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map),
                extent=extent, origin="lower"
            )
            ax[k][panel].set_title("Model Source {} Rendered".format(k))
            ax[k][panel].set_xlim(extent[0], extent[1])
            ax[k][panel].set_ylim(extent[2], extent[3])

            if center is not None and mark_centers:
                ax[k][panel].plot(*center, "wx", mew=1, ms=10)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            _images = observation._model_to_frame(rendered_box)
            ax[k][panel].imshow(
                img_to_rgb(_images, norm=norm, channel_map=channel_map),
                extent=extent, origin="lower"
            )
            ax[k][panel].set_title("Observation".format(k))
            ax[k][panel].set_xlim(extent[0], extent[1])
            ax[k][panel].set_ylim(extent[2], extent[3])

            if center is not None and mark_centers:
                ax[k][panel].plot(*center, "wx", mew=1, ms=10)
            panel += 1

        if show_sed:
            for sed in seds:
                ax[k][panel].plot(sed)
            ax[k][panel].set_xticks(range(len(sed)))
            if hasattr(src.model_frame, "channels") and src.model_frame.channels is not None:
                ax[k][panel].set_xticklabels(src.model_frame.channels)
            ax[k][panel].set_title("SED")
            ax[k][panel].set_xlabel("Channel")
            ax[k][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig
