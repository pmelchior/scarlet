import numpy as np


class Config(object):
    """Blend Configuration

    Attributes
    ----------
    accelerated: bool, default=True
        Whether to use an accelerated proximal agorithm
    update_order: array-like
        Whether to update A before S (`update_order=[0,1]`) or
        S before A (`update_order=[1,0]`) in each step of the fit.
    refine_skip: int, default=10
        How many iterations to skip between refining box sizes/positions.
    center_min_dist: float, default=1e-3
        Minimum change in position required to trigger repositioning a source.
        This prevents sources with positions that are essentially fixed from creating
        small updates that slow down convergence.
    edge_flux_thresh: float, 1.0
        Boxes are resized when the flux at an edge is > `edge_flux_thresh` * `bg_rms`.
    source_sizes: array_like, integer-valued
        Size of the source boxes available when resizing.
        Many of the linear operators are the same for any object with the
        same size boxes. To save memory *scarlet* caches the operators for each
        box size, so this array contains a list of the different available sizes.
        If the requested box size is larger than all available sources sizes then
        the largest element in `source_sizes` is used.
    exact_lipschitz: bool, default=False
        Calculate exact Lipschitz constant in every step (`exact_lipschitz` is `True`)
        or only calculate the Lipschitz constant with significant changes in A,S
        (`exact_lipschitz` is `False`)
    update_model: bool, default=false
        The standard method for fitting A and S is to only update the model
        once per iteration.  If `update_model` is `True` then the model
        is updated twice per iteration (once after the A update and once
        after the S update).
    """
    def __init__(self, accelerated=True, update_order=None, slack=0.2, refine_skip=10, source_sizes=None,
                 center_min_dist=1e-3, edge_flux_thresh=1., exact_lipschitz=False, update_model=False):
        self.accelerated = accelerated
        if update_order is None:
            update_order = [1,0]
        self.update_order = update_order
        self.slack = slack
        self.refine_skip = refine_skip
        self.source_sizes = source_sizes
        self.center_min_dist = center_min_dist
        self.edge_flux_thresh = edge_flux_thresh
        self.exact_lipschitz = exact_lipschitz
        if source_sizes is None:
            source_sizes = np.array([15, 25, 45, 75, 115, 165])
        # Call `self.set_source_sizes` to ensure that all sizes are odd
        self.set_source_sizes(source_sizes)
        self.update_model = update_model

    def set_source_sizes(self, sizes):
        """Set the available source sizes

        To make caching translation and psf convolution possible,
        a predefined set of source sizes is used.
        This method ensures that all of the `source_sizes` are
        odd by adding one to any even sizes.

        Parameters
        ----------
        sizes: array-like
            List of available cached sizes for sources
        """
        if hasattr(sizes, "__iter__"):
            self.source_sizes = np.array(sizes, dtype='int')
            mask = (self.source_sizes % 2 == 0)
            self.source_sizes[mask] += 1
            self.source_sizes.sort()
        else:
            raise NotImplementedError("Source sizes must be list of numbers")

    def find_next_source_size(self, size):
        """Find the smallest source size larger than `size`

        For a source that requires a box of size `size`,
        find the smallest element in `self.source_sizes` that
        is larger than `size`.
        If `size` is larger than every element in `self.source.sizes`,
        use the largest element.

        Parameters
        ----------
        size: int
            Size of the box needed to enclose all of the flux in the source.

        Returns
        -------
        result: int
            Smallest element in `self.source_sizes` that is larger than `size`.
        """
        idx = np.flatnonzero(self.source_sizes >= size)
        if len(idx):
            idx = idx[0]
        else:
            idx = -1
        return self.source_sizes[idx]
