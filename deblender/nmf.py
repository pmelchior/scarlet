from __future__ import print_function, division
from functools import partial
import logging

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import proxmin
from proxmin.nmf import Steps_AS

from . import operators
from .proximal import build_prox_monotonic

logger = logging.getLogger("deblender.nmf")

def convolve_band(P, img):
    """Convolve a single band with the PSF
    """
    if isinstance(P, list) is False:
        return P.dot(img.T).T
    else:
        convolved = np.empty(img.shape)
        for b in range(img.shape[0]):
            convolved[b] = P[b].dot(img[b])
        return convolved

def get_peak_model(A, S, Tx, Ty, P=None, shape=None, k=None):
    """Get the model for a single source
    """
    # Allow the user to send full A,S, ... matrices or matrices for a single source
    if k is not None:
        Ak = A[:, k]
        Sk = S[k]
        if Tx is not None or Ty is not None:
            Txk = Tx[k]
            Tyk = Ty[k]
    else:
        Ak, Sk, Txk, Tyk = A, S.copy(), Tx, Ty
    # Check for a flattened or 2D array
    if len(Sk.shape)==2:
        Sk = Sk.flatten()
    B,N = Ak.shape[0], Sk.shape[0]
    model = np.zeros((B,N))

    # NMF without translation
    if Tx is None or Ty is None:
        if Tx is not None or Ty is not None:
            raise ValueError("Expected Tx and Ty to both be None or neither to be None")
        for b in range(B):
            if P is None:
                model[b] = A[b]*Sk
            else:
                model[b] = Ak[b] * P[b].dot(Sk)
    # NMF with translation
    else:
        if P is None:
            Gamma = Tyk.dot(Txk)
        for b in range(B):
            if P is not None:
                Gamma = Tyk.dot(P[b].dot(Txk))
            model[b] = Ak[b] * Gamma.dot(Sk)
    # Reshape the image into a 2D array
    if shape is not None:
        model = model.reshape((B, shape[0], shape[1]))
    return model

def get_model(A, S, Tx, Ty, P=None, shape=None):
    """Build the model for an entire blend
    """
    B,K,N = A.shape[0], A.shape[1], S.shape[1]
    if len(S.shape)==3:
        N = S.shape[1]*S.shape[2]
        S = S.reshape(K,N)
    model = np.zeros((B,N))

    if Tx is None or Ty is None:
        if Tx is not None or Ty is not None:
            raise ValueError("Expected Tx and Ty to both be None or neither to be None")
        model = A.dot(S)
        if P is not None:
            model = convolve_band(P, model)
        if shape is not None:
            model = model.reshape(B, shape[0], shape[1])
    else:
        for pk in range(K):
            for b in range(B):
                if P is None:
                    Gamma = Ty[pk].dot(Tx[pk])
                else:
                    Gamma = Ty[pk].dot(P[b].dot(Tx[pk]))
                model[b] += A[b,pk]*Gamma.dot(S[pk])
    if shape is not None:
        model = model.reshape(B, shape[0], shape[1])
    return model

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

def prox_likelihood(X, step, Xs=None, j=None, Y=None, W=None, Gamma=None,
                    prox_S=None, prox_A=None):
    if j == 0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, Gamma=Gamma, prox_g=prox_A, W=W)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, Gamma=Gamma, prox_g=prox_S, W=W)

def init_A(B, K, peaks=None, img=None):
    # init A from SED of the peak pixels
    if peaks is None:
        A = np.random.rand(B,K)
    else:
        assert img is not None
        assert len(peaks) == K
        A = np.zeros((B,K))
        for k in range(K):
            px,py = peaks[k]
            A[:,k] = img[:,int(py),int(px)]
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
        for pk, (px,py) in enumerate(peaks):
            S[pk, cy*M+cx] = np.abs(img[:,int(py),int(px)].mean()) + tiny
    return S

def adapt_PSF(psf, B, shape, threshold=1e-2):
    # Simpler for likelihood gradients if psf = const across B
    if len(psf.shape)==2: # single matrix
        return operators.getPSFOp(psf, shape, threshold=threshold)

    P_ = []
    for b in range(B):
        P_.append(operators.getPSFOp(psf[b], shape, threshold=threshold))
    return P_

def get_constraint_op(constraint, shape, K, useNearest=True):
    """Get appropriate constraint operator
    """
    N,M = shape
    if constraint == " ":
        return None
    elif constraint=="m":
        return None
    elif constraint == "M":
        # block diagonal matrix to run single dot operation on all components
        L = operators.getRadialMonotonicOp((N,M), useNearest=useNearest)
        LB = scipy.sparse.block_diag([L for k in range(K)])
        return proxmin.utils.MatrixAdapter(LB, axis=1)
    elif constraint == "S":
        L = operators.getSymmetryOp((N,M))
        LB = scipy.sparse.block_diag([L for k in range(K)])
        return proxmin.utils.MatrixAdapter(LB, axis=1)

    raise ValueError("'constraint' should be in [' ', 'm', 'M', 'S'] but received '{0}'".format(constraint))

def translate_psfs(shape, peaks, B, P, threshold=1e-8):
    # Initialize the translation operators
    K = len(peaks)
    Tx = []
    Ty = []
    cx, cy = int(shape[1]/2), int(shape[0]/2)
    for pk, (px, py) in enumerate(peaks):
        dx = cx - px
        dy = cy - py
        tx, ty, _ = operators.getTranslationOp(dx, dy, shape, threshold=threshold)
        Tx.append(tx)
        Ty.append(ty)

    # TODO: This is only temporary until we fit for dx, dy
    Gamma = []
    for pk in range(K):
        if P is None:
            gamma = [Ty[pk].dot(Tx[pk])]*B
        else:
            gamma = []
            for b in range(B):
                g = Ty[pk].dot(P[b].dot(Tx[pk]))
                gamma.append(g)
        Gamma.append(gamma)
    return Tx, Ty, Gamma

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
            psf_thresh=1e-2,
            monotonicUseNearest=False,
            traceback=False,
            translation_thresh=1e-8,
            prox_A=None,
            prox_S=None,
            steps_g=None,
            steps_g_update='steps_f'):

    # vectorize image cubes
    B,N,M = img.shape
    K = len(peaks)
    if sky is None:
        Y = img.reshape(B,N*M)
    else:
        Y = (img-sky).reshape(B,N*M)
    if weights is None:
        W = weights
    else:
        W = weights.reshape(B,N*M)
    if psf is None:
        P_ = psf
    else:
        P_ = adapt_PSF(psf, B, (N,M), threshold=psf_thresh)
    logger.debug("Shape: {0}".format((N,M)))

    # init matrices
    A = init_A(B, K, img=img, peaks=peaks)
    S = init_S(N, M, K, img=img, peaks=peaks)
    Tx, Ty, Gamma = translate_psfs((N,M), peaks, B, P_, threshold=1e-8)

    # constraints on S: non-negativity or L0/L1 sparsity plus ...
    if prox_S is None:
        from functools import partial
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

    # Load linear constraint operators
    if constraints is not None:



        linear_constraints = {
            " ": proxmin.operators.prox_id,    # do nothing
            "M": proxmin.operators.prox_plus,  # positive gradients
            "S": proxmin.operators.prox_zero,  # zero deviation of mirrored pixels
        }
        if "m" in constraints:
            linear_constraints["m"] = build_prox_monotonic((N,M), prox_chain=prox_S)

        # Proximal Operator for each constraint
        proxs_g = [[proxmin.operators.prox_id], # no additional A constraints (yet)
                   [linear_constraints[c] for c in constraints] # S constraints
        ]
        # Linear Operator for each constraint
        Ls = [
            [None], # none need for A
            [get_constraint_op(c, (N,M), K, useNearest=monotonicUseNearest) for c in constraints]
        ]

    else:
        proxs_g = [proxmin.operators.prox_id] * 2
        Ls = [None] * 2

    logger.debug("proxs_g: {0}".format(proxs_g))
    logger.debug("steps_g: {0}".format(steps_g))
    logger.debug("steps_g_update: {0}".format(steps_g_update))
    logger.debug("Ls: {0}".format(Ls))

    # define objective function with strict_constraints
    f = partial(prox_likelihood, Y=Y, W=W, Gamma=Gamma, prox_S=prox_S, prox_A=prox_A)

    # create stepsize callback, needs max of W
    if W is not None:
        Wmax = W.max()
    else:
        W = Wmax = 1
    steps_f = Steps_AS(Wmax=Wmax)

    # run the NMF with those constraints
    Xs = [A, S]
    res = proxmin.algorithms.glmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, traceback=traceback)

    if not traceback:
        A, S = res
        tr = None
    else:
        [A, S], tr = res

    # create the model and reshape to have shape B,N,M
    model = get_model(A, S, Tx, Ty, P_, (N,M))
    S = S.reshape(K,N,M)

    return A, S, model, P_, Tx, Ty, tr
