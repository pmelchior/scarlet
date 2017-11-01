from __future__ import print_function, division
from functools import partial
import logging
from numbers import Number

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import proxmin
from proxmin.nmf import Steps_AS

from . import operators
from .proximal import build_prox_monotonic, prox_cone

# Set basestring in python 3
try:
    basestring
except NameError:
    basestring = str

logger = logging.getLogger("deblender.nmf")

component_types = ["star", "bulge", "disk", # currently supported types
                   "arm", "bar", "jet", "sfr", "tail"] # currently unsupported types

class Deblend(object):
    """Result of the deblender
    """
    def __init__(self, img, A, S, T, W=None, traceback=None, **parameters):
        """Create a deblender result

        `parameters` is used to store the parameters sent to the glmm algorithm, which makes it
        possible to recreate any variable in a given step (if `traceback` is not `None`).
        """
        self.img = img
        self.B, N, M = img.shape
        self.shape = (N,M)
        self.A = A
        self.S = S
        self.T = T
        self.W = W
        self.traceback = traceback
        self.parameters = parameters

    def get_model(self, k=None, combine=False):
        """Extract a model
        """
        if k is None:
            return get_model(self.A, self.S, self.T.Gamma, self.shape, combine)
        else:
            return get_peak_model(self.A, self.S, self.T.Gamma, self.shape, k)

    def get_history(self, variable="S", param=None, idx=None, it=None):
        """Get the history of a parameter (or all parameters)

        If `param` is `None`, the history of all parameters for A and S are returned.
        Otherwise, only the history of a specific parameter is returned.
        If `idx` is not None, only the
        """
        if it is None:
            it = slice(None,None)
        if variable == "A":
            j = 0
        elif variable == "S":
            j = 1
        else:
            raise ValueError("variable should either be 'A' or 'S'")
        hist = {key:np.array(val)[it] for key,val in self.traceback.history[j].items()}

        if param is not None:
            hist = hist[param]
            if idx is not None:
                hist = hist[idx]
            elif param=="X":
                hist = hist[0]
        return hist

class PeakComponent:
    """Stellar model or a single component of a larger object
    """
    def __init__(self, peak, peak_type, sed=None, morphology=None):
        self.peak = peak
        self.type = peak_type
        self.init_sed = sed
        self.init_morphology = morphology
        self.sed = sed
        self.morphology = morphology
        # Set the indices to refer to this component from the PeakCatalog
        self.cidx = self.peak.component_list.index(self.type)
        pidx = self.peak.index
        if self.peak.index != 0:
            idx = self.peak.parent.indices[self.peak.index-1]
        else:
            idx = 0
        self.index = idx+self.cidx

    @property
    def x(self):
        """Always use the parent x value
        """
        return self.peak.x

    @property
    def y(self):
        """Always use the parent y value
        """
        return self.peak.y

    @property
    def Gamma(self):
        """Always use the parent Gamma operator
        """
        return self.peak.Gamma

    @property
    def Tx(self):
        """Always use the parent Tx operator
        """
        return self.peak.Tx

    @property
    def Ty(self):
        """Always use the parent Ty operator
        """
        return self.peak.Ty

class Peak:
    """Stellar or galaxy source with (potentially) multiple components
    """
    def __init__(self, parent, index, x, y, components=None, Tx=None, Ty=None, Gamma=None):
        self.index = index
        self.parent = parent
        self.x = x
        self.y = y
        self.int_tx = {}
        self.int_ty = {}
        self.Tx = Tx
        self.Ty = Ty
        self.Gamma = Gamma

        if components is not None:
            self.component_list = components
            self.components = {c: PeakComponent(self, c) for c in components}
        else:
            self.component_list = ["bulge"]
            self.components = {"bulge": PeakComponent(self, "bulge")}
        self.component_indices = np.array([comp.index for comp in self])

    def __getitem__(self, idx):
        """Select component of the peak

        `idx` can either a string (the name of the component)
        or a number (the index of the component).
        """
        if isinstance(idx, Number):
            idx = self.component_list[idx]
        return self.components[idx]

class PeakCatalog:
    """A collection of stars and galaxies, each one with multiple components
    """
    def __init__(self, peaks, components=None):
        if components is None:
            components = [None]*len(peaks)
            self.indices = np.arange(len(peaks))+1
        else:
            self.indices = np.cumsum([len(c) for c in components])
        self.peaks = [Peak(self, n, pk[0], pk[1], components[n]) for n, pk in enumerate(peaks)]
        self.component_list = [pk.component_list for pk in self.peaks]

    def __getitem__(self, idx):
        """Select the component, not peak, for the given index
        """
        _idx = np.searchsorted(self.indices, idx, side='right')
        if _idx>0:
            cidx = idx-self.indices[_idx-1]
        else:
            cidx = idx
        peak = self.peaks[_idx]
        return peak[cidx]

    def __len__(self):
        """Total number of components in the peak catalog

        This is different than the total number of peaks, since some peaks
        might have multiple components.
        """
        return np.sum([len(p.component_list) for p in self.peaks])

def get_peak_model(A, S, Gamma, shape=None, k=None):
    """Get the model for a single source
    """
    # Allow the user to send full A,S, ... matrices or matrices for a single source
    if k is not None:
        Ak = A[:, k]
        Sk = S[k]
        Gk = Gamma[k]
    else:
        Ak, Sk = A, S.copy()
        Gk = Gamma
    # Check for a flattened or 2D array
    if len(Sk.shape)==2:
        Sk = Sk.flatten()
    B,N = Ak.shape[0], Sk.shape[0]
    model = np.zeros((B,N))

    for b in range(B):
        model[b] = Ak[b] * Gk[b].dot(Sk)

    # Reshape the image into a 2D array
    if shape is not None:
        model = model.reshape((B, shape[0], shape[1]))
    return model

def get_model(A, S, Gamma, shape=None, combine=True):
    """Build the model for an entire blend

    If `combine` is `False`, then a separate model is built for each peak.
    """
    models = np.array([get_peak_model(A, S, Gamma, shape, k) for k in range(S.shape[0])])
    if combine:
        models = np.sum(models, axis=0)
    return models

def delta_data(A, S, data, Gamma, D, W=1):
    """Gradient of model with respect to A or S
    """
    B,K,N = A.shape[0], A.shape[1], S.shape[1]
    # We need to calculate the model for each source individually and sum them
    model = np.zeros((B,N))
    for k in range(K):
        for b in range(B):
            model[b] += A[b,k]*Gamma[k][b].dot(S[k])
    diff = W*(model-data)

    if D == 'S':
        result = np.zeros((K,N))
        for k in range(K):
            for b in range(B):
                result[k] += A[b,k]*Gamma[k][b].T.dot(diff[b])
    elif D == 'A':
        result = np.zeros((B,K))
        for k in range(K):
            for b in range(B):
                result[b][k] = diff[b].dot(Gamma[k][b].dot(S[k]))
    else:
        raise ValueError("Expected either 'A' or 'S' for variable `D`")
    return result

def prox_likelihood_A(A, step, S=None, Y=None, Gamma=None, prox_g=None, W=1):
    """A single gradient step in the likelihood of A, followed by prox_g.
    """
    return prox_g(A - step*delta_data(A, S, Y, D='A', Gamma=Gamma, W=W), step)

def prox_likelihood_S(S, step, A=None, Y=None, Gamma=None, prox_g=None, W=1):
    """A single gradient step in the likelihood of S, followed by prox_g.
    """
    return prox_g(S - step*delta_data(A, S, Y, D='S', Gamma=Gamma, W=W), step)

def prox_likelihood(X, step, Xs=None, j=None, Y=None, W=None, T=None,
                    prox_S=None, prox_A=None):
    # Only update once per iteration
    if j == 0 and T.fit_positions:
        # Update the translation operators
        A, S = Xs
        models = get_model(A, S, T.Gamma, combine=False)
        T.update_positions(Y, models, A, S, W)

    if j==0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, Gamma=T.Gamma, prox_g=prox_A, W=W)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, Gamma=T.Gamma, prox_g=prox_S, W=W)

def init_A(B, K=None, peaks=None, img=None):
    # init A from SED of the peak pixels
    if peaks is None:
        if K is None:
            raise ValueError("Either K or peaks must be specified")
        A = np.random.rand(B,K)
    else:
        if K is None:
            K = len(peaks)
        assert img is not None
        assert len(peaks) == K
        A = np.zeros((B,K))
        for k in range(K):
            # Check for a garbage collector or source with no flux
            if peaks[k] is None:
                logger.warn("Using random A matrix for peak {0}".format(k))
                A[:,k] = np.random.rand(B)
            else:
                px,py = peaks[k].x, peaks[k].y
                A[:,k] = img[:,int(py),int(px)]
                if peaks[k].type == "disk":
                    disk_shift = np.linspace(-1.5,1.5, B)
                    A[:,k] += disk_shift
                if np.sum(A[:,k])==0:
                    logger.warn("Peak {0} has no flux, using random A matrix".format(k))
                    A[:,k] = np.random.rand(B)
                peaks[k].init_sed = A[:,k]
    # ensure proper normalization
    A = proxmin.operators.prox_unity_plus(A, 0)
    return A

def init_S(N, M, K, peaks=None, img=None):
    cx, cy = int(M/2), int(N/2)
    S = np.zeros((K, N*M))
    if img is None or peaks is None:
        S[:,cy*M+cx] = 1
    else:
        tiny = 1e-10
        for k, peak in enumerate(peaks):
            if peak is None:
                logger.warn("Using random S matrix for peak {0}".format(k))
                S[k,:] = np.random.rand(N)
            else:
                px, py = int(peak.x), int(peak.y)
                flux = np.abs(img[:,py,px].mean()) + tiny
                # TODO: Improve the initialization of the bulge and disk in S
                # Make the disk slightly larger
                if peak.type == "disk":
                    for px in [-1,0,1]:
                        for py in [-1,0,1]:
                            S[k, (cy+py)*M+(cx+px)] = flux - np.max([np.sqrt(px**2+py**2),.1])
                else:
                    S[k, cy*M+cx] = flux
                peaks[k].init_morphology = S[k, cy*M+cx]
    return S

def adapt_PSF(psf, B, shape):
    # Simpler for likelihood gradients if psf = const across B
    if len(psf.shape)==2: # single matrix
        return operators.getPSFOp(psf, shape)

    P_ = []
    for b in range(B):
        P_.append(operators.getPSFOp(psf[b], shape))
    return P_

def L_when_sought(L, Z, seeks):
    K = len(seeks)
    Ls = []
    for i in range (K):
        if seeks[i]:
            Ls.append(L)
        else:
            Ls.append(Z)
    return Ls

def get_constraint_op(constraint, shape, seeks, useNearest=True):
    """Get appropriate constraint operator
    """
    N,M = shape
    if constraint is None or constraint == "c":
        return None
    elif constraint == "M":
        L = operators.getRadialMonotonicOp((N,M), useNearest=useNearest)
        Z = operators.getIdentityOp((N,M))
    elif constraint == "S":
        L = operators.getSymmetryOp((N,M))
        Z = operators.getZeroOp((N,M))
    elif constraint == "X" or constraint == "x":
        cx = int(shape[1]/2)
        L = proxmin.operators.get_gradient_x(shape, cx)
        Z = operators.getIdentityOp((N,M))
    elif constraint == "Y" or constraint == "y":
        cy = int(shape[0]/2)
        L = proxmin.operators.get_gradient_y(shape, cy)
        Z = operators.getIdentityOp((N,M))
    # Create the matrix adapter for the operator
    LB = scipy.sparse.block_diag(L_when_sought(L, Z, seeks))
    adapter =  proxmin.utils.MatrixAdapter(LB, axis=1)
    return adapter

def oddify(shape, truncate=False):
    """Get an odd number of rows and columns
    """
    if shape is None:
        shape = img.shape
    B,N,M = shape
    if N % 2 == 0:
        if truncate:
            N -= 1
        else:
            N += 1
    if M % 2 == 0:
        if truncate:
            M -= 1
        else:
            M += 1
    return B, N, M

def reshape_img(img, new_shape=None, truncate=False, fill=0):
    """Ensure that the image has an odd number of rows and columns
    """
    if new_shape is None:
        new_shape = oddify(img.shape, truncate)

    if img.shape != new_shape:
        B,N,M = img.shape
        _B,_N,_M = new_shape
        if B != _B:
            raise ValueError("The old and new shape must have the same number of bands")
        if truncate:
            _img = img[:,:_N, :_M]
        else:
            if fill==0:
                _img = np.zeros((B,_N,_M))
            else:
                _img = np.empty((B,_N,_M))
                _img[:] = fill
            _img[:,:N,:M] = img[:]
    else:
        _img = img
    return _img

def deblend(img,
            peaks=None,
            components=None,
            constraints=None,
            weights=None,
            psf=None,
            max_iter=1000,
            sky=None,
            l0_thresh=None,
            l1_thresh=None,
            e_rel=1e-3,
            e_abs=0,
            monotonicUseNearest=False,
            traceback=False,
            translation_thresh=1e-8,
            prox_A=None,
            prox_S=None,
            slack = 0.9,
            update_order=None,
            steps_g=None,
            steps_g_update='steps_f',
            truncate=False,
            fit_positions=True,
            txy_diff=0.1,
            max_shift=2,
            txy_thresh=1e-8,
            txy_wait=10,
            txy_skip=10,
            Translation=operators.TxyTranslation,
            A=None,
            S=None,
            smoothness=1
            ):

    # vectorize image cubes
    B,N,M = img.shape

    # Ensure that the image has an odd number of rows and columns
    _img = reshape_img(img, truncate=truncate)
    if _img.shape != img.shape:
        logger.warn("Reshaped image from {0} to {1}".format(img.shape, _img.shape))
        if weights is not None:
            _weights = reshape_img(weights, _img.shape, truncate=truncate)
        if sky is not None:
            _sky = reshape_img(sky, _img.shape, truncate=truncate)
        B,N,M = _img.shape
    else:
        _img = img
        _weights = weights
        _sky = sky

    # Add peak coordinates for each component of a source
    if not isinstance(peaks, PeakCatalog):
        _peaks = PeakCatalog(peaks, components)
    else:
        _peaks = peaks
    K = len(_peaks)
    if sky is None:
        Y = _img.reshape(B,N*M)
    else:
        Y = (_img-_sky).reshape(B,N*M)
    if weights is None:
        W = Wmax = 1
    else:
        W = _weights.reshape(B,N*M)
        Wmax = np.max(W)
    if psf is None:
        P_ = psf
    else:
        P_ = adapt_PSF(psf, B, (N,M))
    logger.debug("Shape: {0}".format((N,M)))

    # init matrices
    if A is None:
        A = init_A(B, K, img=_img, peaks=_peaks)
    if S is None:
        S = init_S(N, M, K, img=_img, peaks=_peaks)
    T = Translation(_peaks, (N,M), B, P_, txy_diff, max_shift,
                    txy_thresh, fit_positions, txy_wait, txy_skip, traceback)

    # constraints on S: non-negativity or L0/L1 sparsity plus ...
    if prox_S is None:
        if l0_thresh is None and l1_thresh is None:
            prox_S = proxmin.operators.prox_plus
        else:
            # L0 has preference
            if l0_thresh is not None:
                if l1_thresh is not None:
                    logger.warn("warning: l1_thresh ignored in favor of l0_thresh")
                prox_S = partial(proxmin.operators.prox_hard, thresh=l0_thresh)
            else:
                prox_S = partial(proxmin.operators.prox_soft_plus, thresh=l1_thresh)

    # Constraint on A: projected to non-negative numbers that sum to one
    if prox_A is None:
        prox_A = proxmin.operators.prox_unity_plus

    # Load linear operator constraints: the g functions
    if constraints is not None:

        # same constraints for every object?
        seeks = {} # component k seeks constraint[c]
        if isinstance(constraints, basestring):
            for c in constraints:
                seeks[c] = [True] * K
        else:
            assert hasattr(constraints, '__iter__') and len(constraints) == K
            for i in range(K):
                if constraints[i] is not None:
                    for c in constraints[i]:
                        if c not in seeks.keys():
                            seeks[c] = [False] * K
                        seeks[c][i] = True

        all_types = "SMcXYxy"
        for c in seeks.keys():
            if c not in all_types:
                    err = "Each constraint should be None or in {0} but received '{1}'"
                    raise ValueError(err.format([cn for cn in all_types], c))

        linear_constraints = {
            "M": proxmin.operators.prox_plus,  # positive gradients
            "S": proxmin.operators.prox_zero,  # zero deviation of mirrored pixels,
            "c": partial(prox_cone, G=operators.getRadialMonotonicOp((N,M), useNearest=monotonicUseNearest).toarray()),
            "X": proxmin.operators.prox_plus, # positive X gradient
            "Y": proxmin.operators.prox_plus, # positive Y gradient
            "x": partial(proxmin.operators.prox_soft, thresh=smoothness), # l1 norm on X gradient
            "y": partial(proxmin.operators.prox_soft, thresh=smoothness), # l1 norm on Y gradient
        }

        # Proximal Operator for each constraint
        proxs_g = [None, # no additional A constraints (yet)
                   [linear_constraints[c] for c in seeks.keys()] # S constraints
                   ]
        # Linear Operator for each constraint
        Ls = [[proxmin.utils.MatrixAdapter(None)], # none need for A
              [get_constraint_op(c, (N,M), seeks[c], useNearest=monotonicUseNearest) for c in seeks.keys()]
              ]

    else:
        proxs_g = [proxmin.operators.prox_id] * 2
        Ls = [None] * 2

    logger.debug("prox_A: {0}".format(prox_A))
    logger.debug("prox_S: {0}".format(prox_S))
    logger.debug("proxs_g: {0}".format(proxs_g))
    logger.debug("steps_g: {0}".format(steps_g))
    logger.debug("steps_g_update: {0}".format(steps_g_update))
    logger.debug("Ls: {0}".format(Ls))

    # define objective function with strict_constraints
    f = partial(prox_likelihood, Y=Y, W=W, T=T, prox_S=prox_S, prox_A=prox_A)

    steps_f = Steps_AS(Wmax=Wmax, slack=slack, update_order=update_order)

    # run the NMF with those constraints
    Xs = [A, S]
    res = proxmin.algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order,
                                  steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs,
                                  traceback=traceback)

    if not traceback:
        A, S = res
        S = S.reshape(K,N,M)
        result = Deblend(_img, A, S, T, W=W, traceback=None, f=f)
    else:
        [A, S], tr = res
        S = S.reshape(K,N,M)
        result = Deblend(_img, A, S, T, W=W, traceback=tr,
            f=f, steps_f=steps_f, proxs_g=proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order,
            steps_g_update=steps_g_update, e_rel=e_rel)
    return result
