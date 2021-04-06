import time
from typing import Dict

import numpy as np
from .measure import measure_blend
from . import settings


def deblend(data: Dict[str, np.ndarray], max_iter: int, e_rel: float):
    """Deblend a single blend

    :param data: The numpy dictionary of data to deblend.
    :param max_iter: The maximum number of iterations
    :param e_rel: relative error
    :return: tuple:
        * `measurements`: The measurements made on the blend and matched model(s)
        * `observation`: The observation data.
        * `sources`: The deblended models.
    """
    import scarlet
    from ..initialization import init_all_sources, set_spectra_to_match

    # Load the sample images
    images = data["images"]
    mask = data["footprint"]
    weights = 1 / data["variance"] * ~mask
    centers = data["centers"]
    psf = scarlet.ImagePSF(data["psfs"])
    filters = settings.filters

    # Initialize the model, frame, observation, and sources
    t0 = time.time()
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))

    model_frame = scarlet.Frame(images.shape, psf=model_psf, channels=filters)

    observation = scarlet.Observation(
        images, psf=psf, weights=weights, channels=filters
    )
    observation.match(model_frame)

    sources, skipped = init_all_sources(
        model_frame,
        centers,
        observation,
        max_components=2,
        min_components=1,
        min_snr=30,
        thresh=1,
        fallback=True,
        silent=True,
        set_spectra=True,
    )

    # Fit the blend
    t1 = time.time()
    blend = scarlet.Blend(sources, observation)
    blend.fit(max_iter, e_rel=e_rel)
    t2 = time.time()

    if hasattr(observation, "log_norm"):
        log_norm = observation.log_norm
    else:
        # TODO: not quite right, also bitrott since observation *has* log_norm
        _weights = observation.weights
        _images = observation.data
        log_sigma = np.zeros(_weights.shape, dtype=_weights.dtype)
        cuts = _weights > 0
        log_sigma[cuts] = np.log(1 / weights[cuts])
        log_norm = (
            np.prod(_images[cuts].shape) / 2 * np.log(2 * np.pi) + np.sum(log_sigma) / 2
        )

    measurements = {
        "init time": (t1 - t0) * 1000,
        "runtime": (t2 - t1) * 1000 / len(sources),
        "iterations": len(blend.loss),
        # log_norm is included in loss, keeping it here for backward compatibility of measurements
        "logL": blend.loss[-1] - log_norm,
        "init logL": blend.loss[0] - log_norm,
        # TODO: adding the number of skipped sources would be helpful
        # "skipped": len(skipped),
    }

    for k in skipped:
        sources.insert(k, scarlet.NullSource(model_frame))

    source_measurements = measure_blend(data, sources, observation.channels)
    for measurement in source_measurements:
        measurement.update(measurements)

    return source_measurements, observation, sources
