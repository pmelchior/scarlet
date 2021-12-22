import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .bbox import Box, overlapped_slices
from .interpolation import get_filter_coords, get_filter_bounds
from .operator import prox_monotonic_mask
from .wavelet import starlet_reconstruction, starlet_transform, get_multiresolution_support


logger = logging.getLogger("scarlet.detect")


def bounds_to_bbox(bounds):
    """Convert the bounds of a Footprint into a Box

    Parameters
    ----------
    bounds: `tuple` of `(bottom, top, left, right)`
        The bounds of the `Footprint`
    """
    return Box(
        (bounds[1]+1-bounds[0], bounds[3]+1-bounds[2]),
        origin=(bounds[0], bounds[2])
    )


def box_intersect(box1, box2):
    """Check if two boxes overlap

    Parameters
    ----------
    box1, box2: `scarlet.bbox.Box`
        The boxes to check for overlap

    Returns
    -------
    overlap: `bool`
        True when the two boxes overlap
    """
    overlap = box1 & box2
    return overlap.shape[0] != 0 and overlap.shape[1] != 0


def footprint_intersect(footprint1, box1, footprint2, box2):
    """Check if two footprints overlap

    Parameters
    ----------
    box1, box2: `scarlet.bbox.Box`
        The boxes of the footprints to check for overlap.
    footprint1, footprint2: `scarlet.detect_pybind11.Footprint`
        The boolean mask for the two footprints.

    Returns
    -------
    overlap: `bool`
        True when the two footprints overlap.
    """
    if not box_intersect(box1, box2):
        return False
    slices1, slices2 = overlapped_slices(box1, box2)
    overlap = footprint1[slices1] * footprint2[slices2]
    return np.sum(overlap) > 0


def draw_box(box, ax, color):
    """Draw a box on an axis

    Parameters
    ----------
    box: `scarlet.bbox.Box`
        The box to draw
    ax: `matplotlib.Axis`
        The axis on which to draw the box
    color: `str`
        The name of the color to use for the box
    """
    rect = patches.Rectangle(
        box.origin[::-1], box.shape[1], box.shape[0],
        linewidth=1, edgecolor=color, facecolor="none")
    ax.add_patch(rect)

def draw_region(region, ax):
    """Draw a QuadTreeRegion in a plot

    Parameters
    ----------
    region: `QuadTreeRegion`
        The region to draw
    ax: `matplotlib.Axis`
        The axis on which to draw the box
    """
    box = region.bbox
    draw_box(box, ax, "r")
    if region.sub_regions is not None:
        for sub in region.sub_regions:
            draw_region(sub, ax)

def draw_footprint_box(footprint, ax):
    """Draw a scarlet Footprint in a plot

    Parameters
    ----------
    footprint: `scarlet.detect_pybind11.Footprint`
        The footprint to draw
    ax: `matplotlib.Axis`
        The axis on which to draw the box
    """
    box = bounds_to_bbox(footprint.bounds)
    draw_box(box, ax, "k")


class QuadTreeRegion:
    """An implementation of a QuadTree that inserts boxes as opposed to points
    """
    def __init__(self, bbox, capacity=5, sub_regions=None, boxes=None, depth=0,
                 detect=None):
        """Initialize a new QuadTreeRegion instance.

        Parameters
        ----------
        bbox: `scarlet.bbox.Box`
            The box that encloses the `QuadTreeRegion`.
        capacity: `int`
            The maximum number of objects contained in a region before
            splitting into smaller regions.
        sub_regions: `list` of `QuadTreeRegion`
            A list of (4) sub-regions contained in this region.
        boxes: `list` of `scarlet.bbox.Box`
            The bounding boxes contained in this `QuadTreeRegion`.
        depth: `int`
            The depth in the full quad tree of this region.
        """
        self.bbox = bbox
        self.sub_regions = sub_regions
        if boxes is None:
            boxes = []
        self.boxes = boxes
        self.capacity = capacity
        # Used for debugging
        self.depth = depth
        self.detect = detect
        self.debug = detect is not None

    def footprint_image(self, bbox=None):
        """Get an image array of all of the footprints in the tree
        """
        boxes = self.query(self.bbox)

        if bbox is None:
            bbox = Box((0,0))
            for box in boxes:
                bbox = bbox | box

        footprint = np.zeros(bbox.shape)
        for box in boxes:
            full, local = overlapped_slices(bbox, box)
            footprint[full] += box.footprint.footprint[local]
        return footprint

    @property
    def peaks(self):
        """Generate a list of peaks contained in the tree
        """
        for box in self.query(self.bbox):
            for peak in box.footprint.peaks:
                yield peak

    def add(self, other_box):
        """Add a box to the region.

        Parameters
        ----------
        other_box: `scarlet.bbox.Box`
            The box to add to the region.
        """
        if box_intersect(self.bbox, other_box):
            # If the region has already been subdivided,
            # pass the new box to its children.
            if self.sub_regions is not None:
                self._add_to_sub_regions(other_box)
                return
            elif self.boxes is None:
                self.boxes = []

            # If the new box keeps the total number of boxes in this
            # region under the maximum capacity, add it to the list
            # of boxes.
            if len(self.boxes) < self.capacity-1:
                self.boxes.append(other_box)
            else:
                # Subdivide this region and pass its contents down to the
                # subregions.
                self.split()
                self.boxes = None
                self._add_to_sub_regions(other_box)

    def add_footprints(self, footprints):
        """Add bounding boxes for a list of scarlet footprints.

        Parameters
        ----------
        footprints: `list` of `scarlet.detect_pybind11.Footprint`
            A list of footprints detected by scarlet.
        """
        for fp in footprints:
            box = bounds_to_bbox(fp.bounds)
            box.footprint = fp
            self.add(box)
        return self

    def split(self):
        """Sub-divide this region into 4 sub-regions.
        """
        height, width = self.bbox.shape
        h2 = height // 2
        w2 = width // 2
        h3 = height - h2
        w3 = width - w2

        if self.debug:
            # It can be useful for error checking to verify that the regions
            # are subdivided as expected.
            fig, ax = plt.subplots()
            ax.imshow(self.detect[2], cmap="Greys")
            ax.set_title(self.depth)
            draw_box(self.bbox, ax, "r")
            for box in self.boxes:
                draw_box(box, ax, "b")

        origin = self.bbox.origin
        self.sub_regions = [
            QuadTreeRegion(
                Box((h2, w2), origin),
                capacity=self.capacity,
                depth=self.depth+1,
            ),
            QuadTreeRegion(
                Box((h3, w2), (origin[0] + h2, origin[1])),
                capacity=self.capacity,
                depth=self.depth+1,
            ),
            QuadTreeRegion(
                Box((h2, w3), (origin[0], origin[1] + w2)),
                capacity=self.capacity,
                depth=self.depth+1,
            ),
            QuadTreeRegion(
                Box((h3, w3), (origin[0] + h2, origin[1] + w2)),
                capacity=self.capacity,
                depth=self.depth+1,
            ),
        ]
        for box in self.boxes:
            self._add_to_sub_regions(box)

    def _add_to_sub_regions(self, other_box):
        """Add a box to all of the sub-regions of this region

        Parameters
        ----------
        other_box: `scarlet.bbox.Box`
            The box to add to the region.
        """
        for region in self.sub_regions:
            region.add(other_box)

    def query(self, other_box=None):
        """Return all of the boxes that overlap with a given box

        Parameters
        ----------
        other_box: `scarlet.bbox.Box`
            The box to use for the search. All boxes in this region or one
            of its sub-regions that overlap with `other_box` will be returned.

        Returns
        -------
        result: `set` of `scarlet.bbox.BoundingBox`
            The set of all boxes that overlap with `other_box`.
            We use a set instead of a list because some boxes may be in
            multiple sub-regions and we only want to have one copy of each.
        """
        if other_box is None:
            other_box = self.bbox
        if self.boxes is not None:
            results = set([box for box in self.boxes if box_intersect(box, other_box)])
        elif self.sub_regions is not None:
            results = set()
            for region in self.sub_regions:
                if box_intersect(region.bbox, other_box):
                    results |= region.query(other_box)
        else:
            results = set()
        return results


class SingleScaleStructure:
    """A structure at a single scale with quadtrees to lookup child boxes
    at different scales.

    Using the terminology from Starck et al. 2011 we refere to a connected
    set of pixels with a common set of peaks at a single scale as a structure.

    Attributes
    ----------
    scale: `int`
        The wavelet scale of this structure.
    footprint: `scarlet.detect_pybind11.Footprint`
        The footprint of this structure at its given scale.
    bbox: `scarlet.bbox.Box`
        The bounding box of this region.
    peaks: `dict`: {`int`, `list` of `scarlet.detect_pybind11.Peak`}
        The dictionary with each wavelet scale as a `key` with lists
        of `Peak`s as values.
    """
    def __init__(self, scale, footprint):
        """Initialize the SingleScaleStructure

        Parameters
        ----------
        scale: `int`
            The wavelet scale of this structure
        footprint: `scarlet.detect_pybind11.Footprint`
            The footprint of this structure at its given scale.
        """
        self.scale = scale
        self.footprint = footprint
        self.bbox = bounds_to_bbox(footprint.bounds)
        self.peaks = {scale: footprint.peaks}
        self._all_peaks = None

    def add_footprint(self, scale, footprint):
        """Add a footprint to the strcuture

        Parameters
        ----------
        scale: `int`
            The scale of the footprint that is added.
        `footprint`: `scarlet.detect_pybind11.Footprint`
            The footprint to be added to the structure.
        """
        if scale not in self.peaks:
            self.peaks[scale] = []
        self.peaks[scale] += footprint.peaks
        # Clear the cached list of all peaks so that it will be regenerated
        self._all_peaks = None

    def add_scale_tree(self, scale, tree):
        """Add all of the footprints from a region at a different scale
        that overlap with this structure.

        Parameters
        ----------
        scale: `int`
            The scale of the tree that is added.
        tree: `QuadTreeRegion`
            The quad tree that is added at scale `scale`.
        """
        for box in tree.query(self.bbox):
            self.add_footprint(scale, box.footprint)
        return self

    @property
    def all_peaks(self):
        """All of the peaks contained in this Structure

        Returns
        -------
        all_peaks: `set`
            The set of all peaks in the structure, including those
            at different scales.
        """
        if self._all_peaks is not None:
            # If the set of peaks has already been generated,
            # return the cached set of peaks.
            return self._all_peaks
        all_peaks = set()
        for scale, peaks in self.peaks.items():
            all_peaks |= set([(peak.x, peak.y) for peak in peaks])
        self._all_peaks = all_peaks
        return self._all_peaks



def get_wavelets(images, variance, scales=3):
    """Calculate wavelet coefficents given a set of images and their variances

    Parameters
    ----------
    images: array-like
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance: array-like
        An array of variances with the same shape as `images`.
    scales: `int`
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.

    Returns
    -------
    coeffs: `numpy.ndarray`
        The array of coefficents with shape `(scales+1, bands, Ny, Nx)`.
    """
    sigma = np.median(np.sqrt(variance), axis=(1,2))
    # Create the wavelet coefficients for the significant pixels
    coeffs = []
    for b, image in enumerate(images):
        logger.debug(f"generating wavelets for band {b}")
        _coeffs = starlet_transform(image, scales=scales)
        M = get_multiresolution_support(image, _coeffs, sigma[b], K=3, epsilon=1e-1, max_iter=20)
        coeffs.append(M * _coeffs)
    return np.array(coeffs)


def get_detect_wavelets(images, variance, scales=3):
    """Get an array of wavelet coefficents to use for detection

    Parameters
    images: array-like
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance: array-like
        An array of variances with the same shape as `images`.
    scales: `int`
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.
    """
    sigma = np.median(np.sqrt(variance))
    # Create the wavelet coefficients for the significant pixels
    detect = np.sum(images, axis=0)
    _coeffs = starlet_transform(detect, scales=scales)
    M = get_multiresolution_support(detect, _coeffs, sigma, K=3, epsilon=1e-1, max_iter=20)
    return M * _coeffs


def get_blend_structures(detect):
    """Generate a set of structures for the 3rd wavelet scale

    This is a convenience function to generate a hierarchy connecting
    all of the footprints at lower scales to the higher scale structures
    that overlap with them.
    """
    from scarlet.detect_pybind11 import get_footprints
    all_footprints = []

    for scale, _detect in enumerate(detect[:-1]):
        footprints = get_footprints(_detect, min_separation=0, min_area=4, thresh=0)
        all_footprints.append(footprints)

    low, middle = all_footprints[:2]
    low_tree = QuadTreeRegion(Box(detect.shape[-2:]), capacity=10).add_footprints(low)
    middle_tree = QuadTreeRegion(Box(detect.shape[-2:]), capacity=10).add_footprints(middle)

    high_structures = [
        SingleScaleStructure(2, fp).add_scale_tree(0, low_tree).add_scale_tree(1, middle_tree)
        for fp in all_footprints[2]
    ]

    return high_structures, middle_tree


def get_peaks(detect=None, images=None, variance=None, bbox=None, scales=3):
    """Detect all of the peaks in the 2nd wavelet scale

    This is not meant to be a permanent solution, as there are some objects
    that don't have a detection on the 2nd wavelet scale, however through
    testing it has been confirmed that this algorithm works better than the
    LSST science pipelines detection algorithm and is a good replacement
    until the hierarchical detection tree can be better understood and
    finalized.

    Parameters
    ----------
    detect: `numpy.ndarray`
        A set of wavelet coefficents used to detect sources.
        If `detect` is `None` then `images` and `variance`must be specified.
    images: `numpy.ndarray`
        The set of 3D images `(band, height, width)` to use for
        creating the wavelet coefficients.
        This is ignored if detect is not `None`.
    variance: `numpy.ndarray`
        The variance of `images`.
        This is ignored if detect is not `None`.
    bbox: `scarlet.bbox.Box`
        The bounding box for the full image.
        If this is `None`, then a bounding box that is the shape of `images`
        with an origin at `(0,0,0)` is used.
    scales: `int`
        The number of wavelet scales to use for creating the detection
        wavelet coefficients.
        This is ignored if detect is not `None`.

    Returns
    -------
    peaks: `list`
        A list of peaks that have been detected at the 2nd wavelet scale.
    """
    if detect is None:
        if images is None or variance is None or bbox is None:
            raise ValueError("Must pass either 'detect' or 'images' and 'variance' and 'bbox'")
        # Get a set of wavelets for detection
        detect = get_detect_wavelets(images, variance, scales=3)

    if bbox is None:
        bbox = Box(detect.shape[1:])
    else:
        bbox = bbox[1:]

    # Detect a hierarchy of structures in the wavelet coefficients
    structures, tree = get_blend_structures(detect)

    # Extract all of the peaks from the structures
    peaks = []
    for box in tree.query(bbox):
        for peak in box.footprint.peaks:
            peaks.append((peak.y, peak.x))
    return peaks
