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
update_order = [1,0]
slack = 0.2
refine_skip=10
source_sizes = [15,25,45,75,115,165]
center_min_dist = 1e-3
edge_flux_thresh=1.
exact_lipschitz=False

import numpy as np
def fix_source_sizes():
    global source_sizes
    source_sizes = np.array(source_sizes, dtype='int')
    mask = (source_sizes % 2 == 0)
    source_sizes[mask] += 1
    source_sizes.sort()

def find_next_source_size(size):
    # find first element not smaller than size
    idx = np.where(source_sizes >= size)
    # if not possible, use largest element
    if len(idx):
        idx = idx[0][0]
    else:
        idx = -1
    return source_sizes[idx]
