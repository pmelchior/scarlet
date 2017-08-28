from __future__ import print_function, division
from functools import partial
import logging

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
    for pk in range(K):
        for b in range(B):
            model[b] += A[b,pk]*Gamma[pk][b].dot(S[pk])
    diff = W*(model-data)

    if D == 'S':
        result = np.zeros((K,N))
        for pk in range(K):
            for b in range(B):
                result[pk] += A[b,pk]*Gamma[pk][b].T.dot(diff[b])
    elif D == 'A':
        result = np.zeros((B,K))
        for pk in range(K):
            for b in range(B):
                result[b][pk] = diff[b].dot(Gamma[pk][b].dot(S[pk]))
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

def init_A(B, K, peaks=None, img=None):
    # init A from SED of the peak pixels
    if peaks is None:
        A = np.random.rand(B,K)
    else:
        assert img is not None
        assert len(peaks) == K
        A = np.zeros((B,K))
        for k in range(K):
            # Check for a garbage collector or source with no flux
            if peaks[k] is None:
                logger.warn("Using random A matrix for peak {0}".format(k))
                A[:,k] = np.random.rand(B)
            else:
                px,py = peaks[k]
                A[:,k] = img[:,int(py),int(px)]
                if np.sum(A[:,k])==0:
                    logger.warn("Peak {0} has no flux, using random A matrix".format(k))
                    A[:,k] = np.random.rand(B)
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
        for pk, peak in enumerate(peaks):
            if peak is None:
                logger.warn("Using random S matrix for peak {0}".format(pk))
                S[pk,:] = np.random.rand(N)
            else:
                px, py = peak
                S[pk, cy*M+cx] = np.abs(img[:,int(py),int(px)].mean()) + tiny
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
        # block diagonal matrix to run single dot operation on all components
        # with seek == True
        L = operators.getRadialMonotonicOp((N,M), useNearest=useNearest)
        Z = operators.getIdentityOp((N,M))
        LB = scipy.sparse.block_diag(L_when_sought(L, Z, seeks))
    elif constraint == "S":
        L = operators.getSymmetryOp((N,M))
        Z = operators.getZeroOp((N,M))
        LB = scipy.sparse.block_diag(L_when_sought(L, Z, seeks))
    elif constraint =="X":
        cx = int(shape[1]/2)
        L = proxmin.operators.get_gradient_x(shape, cx)
        Z = operators.getIdentityOp((N,M))
    elif constraint =="Y":
        cy = int(shape[0]/2)
        L = proxmin.operators.get_gradient_y(shape, cy)
        Z = operators.getIdentityOp((N,M))
    # Create the matrix adapter for the operator
    LB = scipy.sparse.block_diag(L_when_sought(L, Z, seeks))
    return proxmin.utils.MatrixAdapter(LB, axis=1)

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
            Translation=operators.TxyTranslation):

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

    K = len(peaks)
    if sky is None:
        Y = _img.reshape(B,N*M)
    else:
        Y = (_img-_sky).reshape(B,N*M)
    if weights is None:
        W = Wmax = 1,
    else:
        W = _weights.reshape(B,N*M)
        Wmax = np.max(W)
    if psf is None:
        P_ = psf
    else:
        P_ = adapt_PSF(psf, B, (N,M))
    logger.debug("Shape: {0}".format((N,M)))

    # init matrices
    A = init_A(B, K, img=_img, peaks=peaks)
    S = init_S(N, M, K, img=_img, peaks=peaks)
    T = Translation(peaks, (N,M), B, P_, txy_diff, max_shift,
                    txy_thresh, fit_positions, txy_wait, txy_skip)

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

        all_types = "SMcXY"
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
        }

        # Proximal Operator for each constraint
        proxs_g = [None, # no additional A constraints (yet)
                   [linear_constraints[c] for c in seeks.keys()] # S constraints
                   ]
        # Linear Operator for each constraint
        Ls = [[None], # none need for A
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
    res = proxmin.algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs, traceback=traceback)

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
