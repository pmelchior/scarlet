import numpy as np
from astropy.visualization.lupton_rgb import LinearMapping, AsinhMapping
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import MaxNLocator
from .bbox import Box
from .component import Component
from .source import NullSource


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
        0, 8
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
    elif channels == 7:
        channel_map[:, 6] = 2/3.
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


class AsinhAutomaticNorm(AsinhMapping):
    def __init__(self, observation, channel_map=None, noisefactor=0, percentilemax = 98):
        """Create norm that is linear between a factor of the image noise level and the upper percentile of img
        Parameters
        ----------
        observation: `~scarlet.Observation`
            Observation object with weights  
        channel_map: array_like
            Linear mapping with dimensions (3, channels)
        noisefactor: float, default = 1    
            Factor to be multiplied by the negative median noise level to set the lower bound
            below which pixels will be set to 0.
        percentile: float, default = 98
            Upper percentile: Pixel values above will be saturated.
        """
        import numpy.ma as ma 
        if channel_map==None:
            channel_map = channels_to_rgb(observation.data.shape[0]) 
        data = img_to_3channel(observation.data, channel_map=channel_map)
        weights = img_to_3channel(observation.weights, channel_map=channel_map) 
        mask = np.sum(weights, axis=0) == 0 
        ny, nx = mask.shape
        mask = mask.reshape(1, ny, nx)
        mask = np.repeat(mask, 3, axis=0)  
        data = ma.masked_array(data, mask=mask)
        weights = ma.masked_array(weights, mask=mask) 
        vmin = -noisefactor * np.max(np.ma.median(1/np.sqrt(weights), axis=(1,2)))
        vmax = np.max(np.nanpercentile(data, percentilemax,axis=(1,2)))   
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


panel_size = 4.0


def show_likelihood(blend, figsize=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(blend.log_likelihood, **kwargs)
    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("log-Likelihood")
    return fig


def show_observation(
    observation,
    norm=None,
    channel_map=None,
    sky_coords=None,
    show_psf=False,
    add_labels=True,
    figsize=None,
):
    """Plot observation in standardized form.
    """
    panels = 1 if show_psf is False else 2
    if figsize is None:
        figsize = (panel_size * panels, panel_size)
    fig, ax = plt.subplots(1, panels, figsize=figsize)
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    # Mask any pixels with zero weight in all bands
    mask = np.sum(observation.weights, axis=0) == 0
    # if there are no masked pixels, do not use a mask
    if np.all(mask == 0):
        mask = None

    panel = 0
    extent = get_extent(observation.bbox)
    ax[panel].imshow(
        img_to_rgb(observation.data, norm=norm, channel_map=channel_map, mask=mask),
        extent=extent,
        origin="lower",
    )
    ax[panel].set_title("Observation")

    if add_labels:
        assert sky_coords is not None, "Provide sky_coords for labeled objects"

        for k, center in enumerate(sky_coords):
            if hasattr(observation, "get_pixel"):
                center_ = observation.get_pixel(center)
                color = "w" if observation.C > 1 else "r"
            else:
                center_ = center
                color = "w" if observation.data.shape[0] > 1 else "r"
            ax[panel].text(*center_[::-1], k, color=color, ha="center", va="center")

    panel += 1
    if show_psf:
        psf_image = np.zeros(observation.data.shape)

        if observation.psf is not None:
            psf_model = observation.psf.get_model()
            # make PSF as bright as the brightest pixel of the observation
            psf_model *= (
                observation.data.mean(axis=0).max() / psf_model.mean(axis=0).max()
            )
            # insert into middle of "blank" observation
            full_box = Box(psf_image.shape)
            shift = tuple(
                psf_image.shape[c] // 2 - psf_model.shape[c] // 2
                for c in range(full_box.D)
            )
            model_box = Box(psf_model.shape) + shift
            model_box.insert_into(psf_image, psf_model)
            # slices = scarlet.box.overlapped_slices
        ax[panel].imshow(img_to_rgb(psf_image, norm=norm), origin="lower")
        ax[panel].set_title("PSF")

    fig.tight_layout()
    return fig


def show_scene(
    sources,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_residual=False,
    add_labels=True,
    add_boxes=False,
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
    add_label: bool
        Whether each source is labeled with its numerical index in the source list
    add_boxes: bool
        Whether each source box is shown
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
            figsize = (panel_size * panels, panel_size)
        fig, ax = plt.subplots(1, panels, figsize=figsize)
    else:
        columns = int(np.ceil(panels / 2))
        if figsize is None:
            figsize = (panel_size * columns, panel_size * 2)
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(2, columns, n + 1) for n in range(panels)]
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    # Mask any pixels with zero weight in all bands
    if observation is not None:
        mask = np.sum(observation.weights, axis=0) == 0
        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None

    model_frame = observation.model_frame#sources[0].frame
    
    model = np.zeros(model_frame.shape)
    for src in sources:
        model += src.get_model(frame=model_frame)

    panel = 0
    if show_model:
        extent = get_extent(model_frame.bbox)
        ax[panel].imshow(
            img_to_rgb(model, norm=norm, channel_map=channel_map),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model")
        panel += 1

    if show_rendered or show_residual: 
        model = observation.render(model)
        extent = get_extent(observation.bbox)

    if show_rendered:
        norm_ = LinearPercentileNorm(model,percentiles=[1,99.9])

        ax[panel].imshow(
            img_to_rgb(model, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model Rendered")
        panel += 1

    if show_observed:
        norm_ = LinearPercentileNorm(observation.data, percentiles=[1,99.9])

        ax[panel].imshow(
            img_to_rgb(observation.data, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Observation")
        panel += 1

    if show_residual:
        residual = observation.data - model
        norm_ = LinearPercentileNorm(residual)
        ax[panel].imshow(
            img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Residual")
        panel += 1

    for k, src in enumerate(sources):
        if add_boxes:
            panel = 0
            box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}
            if show_model:
                extent = get_extent(src.bbox)
                rect = Rectangle(
                    (extent[0], extent[2]),
                    extent[1] - extent[0],
                    extent[3] - extent[2],
                    **box_kwargs
                )
                ax[panel].add_artist(rect)
                panel = 1
            if observation is not None:
                start, stop = src.bbox.start[-2:][::-1], src.bbox.stop[-2:][::-1]
                points = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
                coords = [
                    observation.get_pixel(model_frame.get_sky_coord(p)) for p in points
                ]
                for panel in range(panel, panels):
                    poly = Polygon(coords, closed=True, **box_kwargs)
                    ax[panel].add_artist(poly)

        if add_labels and hasattr(src, "center") and src.center is not None:
            center = src.center
            panel = 0
            if show_model:
                ax[panel].text(*center[::-1], k, color="w", ha="center", va="center")
                panel = 1
            if observation is not None:
                center_ = observation.get_pixel(model_frame.get_sky_coord(center))
                for panel in range(panel, panels):
                    ax[panel].text(
                        *center_[::-1], k, color="w", ha="center", va="center"
                    )

    fig.tight_layout()
    return fig


def get_extent(bbox):
    return [bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]]


def show_sources(
    sources,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_spectrum=True,
    figsize=None,
    model_mask=None,
    add_markers=True,
    add_boxes=False,
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
    show_spectrum: bool
        Whether source specturm is shown.
        For multi-component sources, spectra are shown separately.
    figsize: matplotlib figsize argument
    model_mask: array
        Mask used to hide pixels in the model only.
    add_markers: bool
        Whether or not to mark the centers of the sources
        with their source number.
    add_boxes: bool
        Whether source boxes are shown
    Returns
    -------
    matplotlib figure
    """
    if show_observed or show_rendered:
        assert (
            observation is not None
        ), "Provide matched observation to show observed frame"

    panels = sum((show_model, show_observed, show_rendered, show_spectrum))
    n_sources = len([src for src in sources if not isinstance(src, NullSource)])
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False)

    marker_kwargs = {"mew": 1, "ms": 10}
    box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    skipped = 0
    for k, src in enumerate(sources):
        # skip NullSources
        if isinstance(src, NullSource):
            skipped += 1
            continue

        model_frame = src.frame

        if hasattr(src, "center") and src.center is not None:
            center = np.array(src.center)[::-1]
        else:
            center = None

        if add_boxes:
            start, stop = src.bbox.start[-2:][::-1], src.bbox.stop[-2:][::-1]
            points = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
            box_coords = [
                observation.get_pixel(model_frame.get_sky_coord(p)) for p in points
            ]

        # model in its bbox
        panel = 0
        model = src.get_model()

        if show_model:
            # Show the unrendered model in it's bbox
            extent = get_extent(src.bbox)
            ax[k-skipped][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {}".format(k))
            if center is not None and add_markers:
                ax[k-skipped][panel].plot(*center, "wx", **marker_kwargs)
            panel += 1

        # model in observation frame
        if show_rendered:
            # Center and show the rendered model
            model_ = src.get_model(frame=model_frame)
            model_ = observation.render(model_)
            extent = get_extent(observation.bbox)
            ax[k-skipped][panel].imshow(
                img_to_rgb(model_, norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {} Rendered".format(k))

            if center is not None and add_markers:
                center_ = observation.get_pixel(model_frame.get_sky_coord(center))
                ax[k-skipped][panel].plot(*center_, "wx", **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k-skipped][panel].add_artist(poly)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            _images = observation.data
            ax[k-skipped][panel].imshow(
                img_to_rgb(_images, norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Observation".format(k))
            if center is not None and add_markers:
                center_ = observation.get_pixel(model_frame.get_sky_coord(center))
                ax[k-skipped][panel].plot(*center_, "wx", **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k-skipped][panel].add_artist(poly)
            panel += 1

        if show_spectrum:
            # needs to be evaluated in the source box to prevent truncation
            if hasattr(src, "__iter__") and isinstance(src[0], Component):
                spectra = []
                for component in src:
                    model_ = component.get_model()
                    spectra.append(model_.sum(axis=(1, 2)))
            else:
                spectra = [model.sum(axis=(1, 2))]

            for spectrum in spectra:
                ax[k-skipped][panel].plot(spectrum)
            ax[k-skipped][panel].set_xticks(range(len(spectrum)))
            if hasattr(src.frame, "channels") and src.frame.channels is not None:
                ax[k-skipped][panel].set_xticklabels(src.frame.channels)
            ax[k-skipped][panel].set_title("Spectrum")
            ax[k-skipped][panel].set_xlabel("Channel")
            ax[k-skipped][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig


def show_scarlet2_scene(
    sources,
    observation=None,
    norm=None,
    model=None,
    frame=None,
    channel_map=None,
    show_observed=False,
    show_rendered=False,
    show_residual=False,
    add_labels=True,
    add_boxes=False,
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
    add_label: bool
        Whether each source is labeled with its numerical index in the source list
    add_boxes: bool
        Whether each source box is shown
            figsize: matplotlib figsize argument
    linear: bool
        Whether or not to display the scene in a single line (`True`) or        on multiple lines (`False`).
    Returns
    -------
    matplotlib figure
    """
    if show_observed or show_rendered or show_residual:
        assert (
            observation is not None
        ), "Provide matched observation to show observed frame"

    panels = sum((show_observed, show_rendered, show_residual))
    if linear:
        if figsize is None:
            figsize = (panel_size * panels, panel_size)
        fig, ax = plt.subplots(1, panels, figsize=figsize)
    else:
        columns = int(np.ceil(panels / 2))
        if figsize is None:
            figsize = (panel_size * columns, panel_size * 2)
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(2, columns, n + 1) for n in range(panels)]
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    # Mask any pixels with zero weight in all bands
    if observation is not None:
        mask = np.sum(observation.weights, axis=0) == 0
        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None

    model_frame = frame#sources[0].frame
    #model = np.zeros(model_frame.shape)
    #for src in sources:
    #    model += src.get_model(frame=model_frame)

    panel = 0
    extent = get_extent(frame.bbox)

    if show_rendered:
        norm_ = LinearPercentileNorm(observation.data,percentiles=[1,99])

        ax[panel].imshow(
            img_to_rgb(model, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model Rendered")
        panel += 1

    if show_observed:
        norm_ = LinearPercentileNorm(observation.data, percentiles=[1,99])

        ax[panel].imshow(
            img_to_rgb(observation.data, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Observation")
        panel += 1

    if show_residual:
        residual = observation.data - model
        norm_ = LinearPercentileNorm(residual)
        ax[panel].imshow(
            img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Residual")
        panel += 1
    
    for k, src in enumerate(sources):
        if add_labels and hasattr(src, "center") and src.center is not None:
            center = src.center
            panel = 0
            if observation is not None:
                center_ = frame.get_pixel(model_frame.get_sky_coord(center))
                for panel in range(panel, panels):
                    ax[panel].text(
                        *center_[::-1], k, color="w", ha="center", va="center"
                    )

    fig.tight_layout()
    return fig
