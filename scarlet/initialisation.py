from .source import *
from .interpolation import interpolate_observation
from .observation import Observation
from .wavelet import Starlet

def build_initialisation_coadd(observations, wavelet = False):
    """Build a channel weighted coadd to use for source detection

    Parameters
    ----------
    observations: `~scarlet.observation.Observation`
        Observation to use for the coadd.

    Returns
    -------
    detect: array
        2D image created by weighting all of the channels by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]
    # The observation that lives in the same plane as the frame
    loc = np.where([type(obs) is Observation for obs in observations])
    # If more than one element is an `Observation`, then pick the first one as a reference (arbitrary)
    obs_ref = observations[loc[0][0]]

    n_images = 0
    coadd = 0
    jacobian = 0
    weights = 0
    for obs in observations:
        try:
            weights = np.array([w[w > 0].mean() for w in obs.weights])
        except:
            raise AttributeError(
                "Observation.weights missing! Please set inverse variance weights"
            )

        if obs is obs_ref:
            if wavelet is True:
                images = Starlet(obs.images).filter()
            else:
                images = obs.images
        else:
            #interpolate low-res to reference resolution
            images = interpolate_observation(obs, obs_ref.frame, wave_filter = wavelet)
        n_images += len(images)
        if wavelet is True:
            coadd += np.sum(images / np.sum(images, axis=(-2,-1))[:, None, None], axis = 0)
        else:
            # Weighted coadd
            coadd += (images * weights[:, None, None]).sum(axis = (0))
            jacobian += weights.sum()

    if wavelet is True:
        coadd /= np.max(coadd)
        bg_cutoff = 0
        return coadd, bg_cutoff
    coadd /= jacobian
    # thresh is multiple above the rms of detect (weighted variance across channels)
    bg_cutoff = np.sqrt((weights ** 2).sum()) / jacobian
    return coadd, bg_cutoff

def initialise(
             frame,
             observations,
             sky_coords,
             sources,
             wavelet = False,
             coadd_sed = True,
):
    """Function that initialises the sources across observations

    Attributes
    ----------
    frame: `~scarlet.Frame`
            The model's frame
    observations: list of `scarlet.Observation` or `scarlet.LowResObservation` objects
        the observation for which sources have to be initialise.
    sky_coords: list
        list of coordinates othe sources (in Ra-Dec).
        There should be the same number of coordinates as there are sources.
    sources: list of Sources
        list of scarlet.*Source that give the type of sources to initialise
        for each coordinate in sky_coords. If one source is given, this source
        will be used for all coordinates.
    """
    # Make sure sources is iterable. (if only one source user could still provide an iterable or not)
    try:
        iter(sources)
    except TypeError:
        sources = [sources]
    # Assert there is more than one set of coordinates packaged in a list
    try:
        iter(sky_coords[0])
    except TypeError:
        sky_coords = [sky_coords]

    try:
        iter(observations)
    except TypeError:
        observations = [observations]


    if len(sources) != 1:
        assert len(sky_coords) == len(sources), \
            "sky_coords should have the same length as sources, unless sources is of length 1. " \
            f"Received len(sky_coords) = {len(sky_coords)}, len(source) = {len(sources)}."
    # Build coadds:
    if coadd_sed is False:
        coadd, bg_cutoff= build_initialisation_coadd(observations, wavelet = wavelet)
    else:
        coadd, bg_cutoff = None, None
    source_list = []
    for i, coord in enumerate(sky_coords):
        if len(sources) == 1:
            source = sources[0]
        else:
            source = sources[i]
        if type(source) in (Point, Random):
            source_list.append(source(frame,
                                          coord,
                                          observations,
                                          ))
        else:
            source_list.append(source(frame,
                                  coord,
                                  observations,
                                  coadd=coadd,
                                  bg_cutoff=bg_cutoff
                                  ))

    return source_list