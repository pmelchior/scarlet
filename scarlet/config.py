import numpy as np

class Config(object):
    """
        refine_skip: int, default=10
            How many iterations to skip before refining box sizes/positions
        center_min_dist: float, default=1e-3
            Minimum change is position required to trigger repositioning a source
        edge_flux_thresh: float, 1.0
            Boxes are resized when flux at an edge is > `edge_flux_thresh` * `bg_rms`
        source_sizes: array_like, integer-valued
            Size of the source boxes available when resizing
        exact_lipschitz: bool, default=False
            Calculate exact Lipschitz constant in every step or only calculate the Lipschitz
            constant with significant changes in A,S
    """
    def __init__(self):
        self.update_order = [1,0]
        self.slack = 0.2
        self.refine_skip=10
        self.source_sizes = np.array([15,25,45,75,115,165]) # int, odd, sorted
        self.center_min_dist = 1e-3
        self.edge_flux_thresh=1.
        self.exact_lipschitz=False

    def set_source_sizes(self, sizes):
        if hasattr(sizes, "__iter__"):
            self.source_sizes = np.array(sizes, dtype='int')
            mask = (self.source_sizes % 2 == 0)
            self.source_sizes[mask] += 1
            self.source_sizes.sort()
        else:
            raise NotImplementedError("Source sizes must be list of numbers")

    def find_next_source_size(self, size):
        # find first element not smaller than size
        # if not possible, use largest element
        idx = np.flatnonzero(self.source_sizes >= size)
        if len(idx):
            idx = idx[0]
        else:
            idx = -1
        return self.source_sizes[idx]
