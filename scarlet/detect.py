import numpy as np
from scipy.spatial import cKDTree

from .bbox import Box
from .interpolation import get_filter_coords, get_filter_bounds
from .operator import prox_monotonic_tree
from .wavelet import starlet_reconstruction, starlet_transform


def bounds_to_bbox(bounds):
    return Box(
        (bounds[1]+1-bounds[0], bounds[3]+1-bounds[2]),
        origin=(bounds[0], bounds[2]))


def merge_peaks(all_peaks, min_separation=3):
    peaks = all_peaks[0]
    tree = cKDTree(peaks)
    for _peaks in all_peaks[1:]:
        for peak in _peaks:
            dist, _ = tree.query(peak)
            if dist > min_separation:
                peaks.append(peak)
        # Rebuild the tree for each scale
        tree = cKDTree(peaks)
    return peaks


def initialize_peaks(detect, min_area=4, min_separation=3):
    from scarlet.detect_pybind11 import get_footprints

    all_peaks = []
    for scale, _detect in enumerate(detect[:-1]):
        footprints = get_footprints(_detect, min_separation=min_separation, min_area=min_area, thresh=0)
        peaks = [peak for footprint in footprints for peak in footprint.peaks]
        all_peaks.append(peaks)
    peaks = merge_peaks(all_peaks[:2], min_separation=min_separation)
    return peaks


def peaks_to_objects(peaks, wavelets, peak_search_radius):
    structures = {}

    for scale, _wavelets in enumerate(wavelets):
        for peak in peaks:
            structure = MonotonicStructure.from_wavelets(_wavelets, peak, scale, peak_search_radius)
            peak = tuple(peak)
            if peak not in structures.keys():
                structures[peak] = []
            structures[peak].append(structure)

    objects = []
    for peak, structures in structures.items():
        obj = MonotonicObject(structures, peak, peak_search_radius)
        objects.append(obj)
    return objects


def initialize_objects(detect, wavelets, peaks=None, min_area=4, peak_search_radius=2):
    if peaks is None:
        peaks = initialize_peaks(detect, min_area, min_separation=peak_search_radius)
    objects = peaks_to_objects(peaks, wavelets, peak_search_radius)
    return objects


class MonotonicStructure:
    def __init__(self, model, bbox, peak, scale, peak_search_radius):
        self.model = model
        self.bbox = bbox
        self.peak = peak
        self.scale = scale
        self.peak_radius = peak_search_radius
        self.z = model.copy()
        self.t = 1

    @staticmethod
    def from_wavelets(wavelets, peak, scale, peak_search_radius):
        _, model, bounds = prox_monotonic_tree(wavelets, 0, peak, peak_search_radius, max_iter=20)
        bbox = bounds_to_bbox(bounds)
        return MonotonicStructure(model[bbox.slices], bbox, peak, scale, peak_search_radius)

    def update(self, grad):
        """"# Beck-Teboulle proximal gradient (with Nesterov acceleration)
        step = 0.01
        y = -step * grad
        y[self.bbox.slices] += self.z

        # Set the new bounding box and model
        _, model, bounds = prox_monotonic_tree(y, 0, self.peak, self.peak_radius, max_iter=10)
        old_bbox = self.bbox
        self.bbox = bounds_to_bbox(bounds)
        model = model[self.bbox.slices]

        # The new model might be in a different sized box,
        # so we need to project the old model into the new
        # model box
        old_model = np.zeros(model.shape, model.dtype)
        new_slices, old_slices = overlapped_slices(self.bbox, old_bbox)
        old_model[new_slices] = self.model[old_slices]

        # Update the helper parameters
        t = (1 + np.sqrt(4 * self.t ** 2 + 1)) * 0.5
        lam = 1 + (self.t - 1) / t
        self.z = old_model + lam * (model - old_model)
        self.t = t
        self.model = model

        """
        # backup the unmodified gradient
        grad = grad.copy()
        oldSlices = self.bbox.slices
        backup_grad = grad[oldSlices].copy()
        grad[self.bbox.slices] += self.model

        # Set the new bounding box and model
        _, model, bounds = prox_monotonic_tree(grad, 0, self.peak, self.peak_radius, max_iter=10)
        self.bbox = bounds_to_bbox(bounds)
        self.model = model[self.bbox.slices]

        # restore the gradient
        grad[oldSlices] = backup_grad


class MonotonicObject:
    def __init__(self, structures, peak, peak_search_radius):
        self.structures = structures
        self.peak = peak
        self.peak_search_radius = peak_search_radius

    @property
    def bbox(self):
        bbox = self.structures[0].bbox
        for struct in self.structures:
            bbox &= struct.bbox
        return bbox

    def image(self, shape):
        bbox = self.bbox
        model = np.zeros((len(self.structures),) + shape)
        for scale, struct in enumerate(self.structures):
            if struct.model.size > 0:
                model[(scale,) + struct.bbox.slices] = struct.model
        return starlet_reconstruction(model)

    def update(self, grad):
        for scale, structure in enumerate(self.structures):
            structure.update(grad[scale])


def get_model(objects, shape):
    model = np.zeros(shape)
    for obj in objects:
        for scale, struct in enumerate(obj.structures):
            # assert np.min(struct.model) >= 0
            model[(scale,) + struct.bbox.slices] += struct.model
    return model


def get_z_model(objects, shape):
    model = np.zeros(shape)
    for obj in objects:
        for scale, struct in enumerate(obj.structures):
            # assert np.min(struct.model) >= 0
            model[(scale,) + struct.bbox.slices] += struct.z
    return model


def convolve_model(model, filters):
    from scarlet.operators_pybind11 import apply_filter

    result = np.empty(model.shape, dtype=model.dtype)
    # No need to convolve the first two scales
    result[0] = model[0]

    for scale in range(1, len(model)):
        phi, bounds = filters[scale]
        apply_filter(
            model[scale],
            phi[0].reshape(-1),
            bounds[0][0],
            bounds[0][1],
            bounds[0][2],
            bounds[0][3],
            result[scale],
        )

        apply_filter(
            result[scale].copy(),
            phi[1].reshape(-1),
            bounds[1][0],
            bounds[1][1],
            bounds[1][2],
            bounds[1][3],
            result[scale],
        )

    return result


def wavelet_deblend(wavelets, objects, max_iter=100, stepsize=0.2):
    M = wavelets != 0

    filters = {}

    for scale in range(1, len(wavelets)):
        x1 = 2 ** scale
        x2 = 2 ** (scale + 1)
        x = np.linspace(-(x2 - 1), x2 - 1, 2 * x2 - 1)
        y = (np.abs(x - x2) ** 3 - 4 * np.abs(x - x1) ** 3 + +6 * np.abs(x) ** 3 - 4 * np.abs(x + x1) ** 3 + np.abs(
            x + x2) ** 3) / 12
        phi = y / np.sum(y)
        phi_x = phi[None, :]
        phi_y = phi_x.T
        coords = get_filter_coords(phi_x)
        bounds_x = get_filter_bounds(coords.reshape(-1, 2))
        coords = get_filter_coords(phi_y)
        bounds_y = get_filter_bounds(coords.reshape(-1, 2))
        filters[scale] = ((phi_y, phi_x), (bounds_y, bounds_x))

    model = get_model(objects, wavelets.shape)

    # model = np.zeros(wavelets.shape, dtype=wavelets.dtype)

    for it in range(max_iter):
        _image = starlet_reconstruction(model)
        _starlets = starlet_transform(_image, scales=len(model) - 1)
        residual = M * (wavelets - _starlets)
        residual = convolve_model(residual, filters)

        for obj in objects:
            obj.update(residual*stepsize)

        #model = get_z_model(objects, wavelets.shape)
        model = get_model(objects, wavelets.shape)
        model = convolve_model(model, filters)
    model = get_model(objects, wavelets.shape)
    model = convolve_model(model, filters)
    return model
