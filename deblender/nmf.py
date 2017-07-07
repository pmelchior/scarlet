from __future__ import print_function, division
from functools import partial
import logging

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from proxmin import proximal
from proxmin.algorithms import glmm

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

def grad_likelihood_A(A, allX, xidx, data, Gamma=None, W=1, **kwargs):
    """A single gradient step in the likelihood of A

    Used with proxmin.proximal.prox_likelihood
    """
    _, S = allX
    return delta_data(A, S, data, D='A', Gamma=Gamma, W=W)

def grad_likelihood_S(S, allX, xidx, data, Gamma=None, W=1, **kwargs):
    """A single gradient step in the likelihood of S

    Used with proxmin.proximal.prox_likelihood
    """
    A, _ = allX
    #logger.debug("grad_S:\n{0}".format(np.max(delta_data(A, S, data, D='S', Gamma=Gamma, W=W))))
    return delta_data(A, S, data, D='S', Gamma=Gamma, W=W)

def dot_components(C, X, axis=0, transpose=False):
    """Apply a linear constraint C to each peak in X
    """
    K = X.shape[axis]

    if axis == 0:
        if not transpose:
            CX = [C.dot(X[k]) for k in range(K)]
        else:
            CX = [C.T.dot(X[k]) for k in range(K)]
    if axis == 1:
        if not transpose:
            CX = [C.dot(X[:,k]) for k in range(K)]
        else:
            CX = [C.T.dot(X[:,k]) for k in range(K)]
    return np.stack(CX, axis=axis)

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
    A = proximal.prox_unity_plus(A, 0)
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

def get_constraint_op(constraint, shape, useNearest=True):
    """Get appropriate constraint operator
    """
    N,M = shape
    if constraint == " " or constraint=="m":
        return scipy.sparse.identity(N*M)
    elif constraint == "M":
        return operators.getRadialMonotonicOp((N,M), useNearest=useNearest)
    elif constraint == "S":
        return operators.getSymmetryOp((N,M))
    raise ValueError("'constraint' should be in [' ', 'M', 'S'] but received '{0}'".format(constraint))

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
            strict_constraints=None,
            constraints=None,
            weights=None,
            psf=None,
            max_iter=1000,
            sky=None,
            l0_thresh=None,
            l1_thresh=None,
            gradient_thresh=0,
            e_rel=1e-3,
            psf_thresh=1e-2,
            monotonicUseNearest=False,
            algorithm="GLMM",
            als_max_iter=50,
            min_iter=10,
            step_beta=1.,
            step_g=None,
            traceback=False,
            convergence_func=None,
            translation_thresh=1e-8,
            monotonic_thresh=0,
            show=False):

    # vectorize image cubes
    B,N,M = img.shape
    K = len(peaks)
    if sky is None:
        data = img.reshape(B,N*M)
    else:
        data = (img-sky).reshape(B,N*M)
    if weights is None:
        weights = 1
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

    # Set the proximal operator as the likelhood for A,
    # projected to non-negative numbers that sum to one
    _prox_A = proximal.prox_unity_plus
    prox_A = partial(proximal.prox_likelihood, grad_likelihood=grad_likelihood_A,
                     data=data, prox_g=_prox_A, W=W, Gamma=Gamma, xidx=0)

    # S: non-negativity or L0/L1 sparsity plus ...
    if l0_thresh is None and l1_thresh is None:
        _prox_S = proximal.prox_plus
    else:
        # L0 has preference
        if l0_thresh is not None:
            if l1_thresh is not None:
                logger.warn("weightsarning: l1_thresh ignored in favor of l0_thresh")
            _prox_S = partial(proximal.prox_hard, thresh=l0_thresh)
        else:
            _prox_S = partial(proximal.prox_soft_plus, thresh=l1_thresh)
    if strict_constraints is not None:
        for c in strict_constraints[::-1]:
            if c=="M": # Monotonicity
                _prox_S = build_prox_monotonic((N,M), prox_chain=_prox_S, thresh=monotonic_thresh)
    prox_S = partial(proximal.prox_likelihood, grad_likelihood=grad_likelihood_S,
                     data=data, prox_g=_prox_S, W=W, Gamma=Gamma, xidx=1)

    # Load linear constraint operators
    if constraints is not None:
        linear_constraints = {
            " ": proximal.prox_id,    # do nothing
            "M": partial(proximal.prox_min, l=gradient_thresh), # positive gradients
            "S": proximal.prox_zero,   # zero deviation of mirrored pixels
        }
        if "m" in constraints:
            linear_constraints["m"] = build_prox_monotonic((N,M), prox_chain=prox_S)
        # Proximal Operator for each constraint
        all_prox_g = [
            _prox_A, # No A constraints
            [linear_constraints[c] for c in constraints] # S constraints
        ]
        # Linear Operator for each constraint
        all_constraints = [
            None, # No A constraints
            [get_constraint_op(c, (N,M), useNearest=monotonicUseNearest) for c in constraints]
        ]

        if step_g is None:
            # weightseight of the linear operator (to test for convergence)
            all_step_g = None
            all_constraint_norms = [1, np.array([scipy.sparse.linalg.norm(C) for C in all_constraints[1]])]
            logger.debug("Constraint norms for S (intensity) matrix: {0}".format(all_constraint_norms[1]))
        else:
            all_step_g = step_g
            all_constraint_norms = None
            logger.debug("Constraint steps for S (intensity) matrix: {0}".format(all_step_g))
    else:
        all_prox_g = [None, None]
        all_step_g = None
        all_constraints = None
        all_constraint_norms = None

    logger.debug("prox_g: {0}".format(all_prox_g))
    logger.debug("all_step_g: {0}".format(all_step_g))
    logger.debug("all_constraints: {0}".format(all_constraints))
    # run the NMF with those constraints
    if algorithm is not None and algorithm.lower() == "glmm":
        [A, S], errors, history = glmm(allX=[A,S], all_prox_f=[prox_A, prox_S], all_prox_g=all_prox_g,
                                       all_constraints=all_constraints, max_iter=max_iter,
                                       e_rel=e_rel, step_beta=step_beta, weights=weights,
                                       all_step_g=all_step_g, all_constraint_norms=all_constraint_norms,
                                       traceback=traceback,
                                       convergence_func=convergence_func, min_iter=min_iter,
                                       dot_components=dot_components)
    else:
        raise NotImplementedError("Currently on glmm is supported")
        # TODO: Either implement ADMM and SDMM or remove the following code
        [A, S], errors, history = als(allX=[A,S], all_prox_f=[prox_A, prox_S], all_prox_g=all_prox_g,
                                      all_constraints=all_constraints, max_iter=max_iter,
                                      e_rel=e_rel, step_beta=step_beta, weights=weights,
                                      all_step_g=all_step_g, all_constraint_norms=all_constraint_norms,
                                      traceback=traceback,
                                      convergence_func=convergence_func,
                                      als_max_iter=als_max_iter, algorithms=algorithm, min_iter=min_iter,
                                      dot_components=dot_components)

    # create the model and reshape to have shape B,N,M
    model = get_model(A, S, Tx, Ty, P_, (N,M))

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(model[0])

    S = S.reshape(K,N,M)

    return A, S, model, P_, Tx, Ty, errors
