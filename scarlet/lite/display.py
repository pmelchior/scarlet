import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .. import display


def _add_markers(src, extent, ax, add_markers, add_boxes, marker_kwargs, box_kwargs):
    if add_markers and hasattr(src, "center") and src.center is not None:
        center = np.array(src.center)[::-1]
        ax.plot(*center, "wx", **marker_kwargs)

    if add_boxes:

        rect = Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            **box_kwargs
        )
        ax.add_artist(rect)


def show_sources(
    blend,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_spectrum=True,
    figsize=None,
    model_mask=True,
    add_markers=True,
    add_boxes=False,
    use_flux=False,
):
    panel_size = display.panel_size
    observation = blend.observation
    sources = blend.sources
    panels = sum((show_model, show_observed, show_rendered, show_spectrum))
    n_sources = len([src for src in sources if not src.is_null])
    bbox = observation.bbox
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False)

    marker_kwargs = {"mew": 1, "ms": 10}
    box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    skipped = 0
    for k, src in enumerate(blend.sources):
        if src.is_null:
            skipped += 1
            continue
        if use_flux:
            src_box = src.flux_box
        else:
            src_box = src.bbox

        extent = display.get_extent(src_box)

        # model in its bbox
        panel = 0
        model = src.get_model(use_flux=use_flux)

        if show_model:
            if model_mask:
                _model_mask = np.max(model, axis=0) <= 0
            # Show the unrendered model in it's bbox
            ax[k-skipped][panel].imshow(
                display.img_to_rgb(model, norm=norm, channel_map=channel_map, mask=_model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {}".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, False, marker_kwargs, box_kwargs)
            panel += 1

        # model in observation frame
        if show_rendered:
            # Center and show the rendered model
            model_ = src.get_model(bbox=bbox)
            model_ = observation.render(model_)
            ax[k-skipped][panel].imshow(
                display.img_to_rgb(model_, norm=norm, channel_map=channel_map),
                extent=display.get_extent(observation.bbox),
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {} Rendered".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, add_boxes, marker_kwargs, box_kwargs)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            _images = observation.data
            ax[k-skipped][panel].imshow(
                display.img_to_rgb(_images, norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Observation".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, add_boxes, marker_kwargs, box_kwargs)
            panel += 1

        if show_spectrum:
            spectra = [model.sum(axis=(1, 2))]

            for spectrum in spectra:
                ax[k-skipped][panel].plot(spectrum)
            ax[k-skipped][panel].set_xticks(range(len(spectrum)))
            ax[k-skipped][panel].set_title("Spectrum")
            ax[k-skipped][panel].set_xlabel("Band")
            ax[k-skipped][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig