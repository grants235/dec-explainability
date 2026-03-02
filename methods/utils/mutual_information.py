"""
Mutual Information Estimation Utilities.

Used for measuring discriminative affordance and information-theoretic
quantities in means-end decomposition analysis.
"""

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def knn_mutual_information(X, Y, k=5):
    """
    Estimate mutual information I(X; Y) using k-NN estimator (Kraskov et al., 2004).

    Args:
        X: (n, d_x) array
        Y: (n, d_y) array
        k: number of nearest neighbors

    Returns:
        mi: estimated mutual information (nats)
    """
    n = len(X)
    assert len(Y) == n

    # Joint space
    XY = np.hstack([X, Y])

    # Find k-NN in joint space (Chebyshev distance)
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev', algorithm='kd_tree')
    nn_joint.fit(XY)
    distances, _ = nn_joint.kneighbors(XY)
    eps = distances[:, k]  # distance to k-th neighbor (excluding self)

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric='chebyshev', algorithm='kd_tree')
    nn_x.fit(X)
    nx = np.array([len(nn_x.radius_neighbors([[x]], radius=e - 1e-15, return_distance=False)[0])
                   for x, e in zip(X, eps)])

    nn_y = NearestNeighbors(metric='chebyshev', algorithm='kd_tree')
    nn_y.fit(Y)
    ny = np.array([len(nn_y.radius_neighbors([[y]], radius=e - 1e-15, return_distance=False)[0])
                   for y, e in zip(Y, eps)])

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(0.0, mi)


def entropy_discrete(labels):
    """
    Compute Shannon entropy of a discrete distribution.

    Args:
        labels: 1D array of integer class labels

    Returns:
        H: entropy in nats
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-12))


def conditional_entropy_discrete(labels_y, labels_x):
    """
    Compute conditional entropy H(Y|X) for discrete X and Y.

    Args:
        labels_y: 1D array of Y values
        labels_x: 1D array of X values (conditioning variable)

    Returns:
        H_Y_given_X: conditional entropy in nats
    """
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)
    unique_x = np.unique(labels_x)
    total = len(labels_y)
    H = 0.0
    for x in unique_x:
        mask = (labels_x == x)
        p_x = mask.sum() / total
        H += p_x * entropy_discrete(labels_y[mask])
    return H


def mutual_information_discrete(labels_x, labels_y):
    """
    Compute mutual information I(X;Y) for discrete variables.

    Returns:
        MI in nats
    """
    return entropy_discrete(labels_y) - conditional_entropy_discrete(labels_y, labels_x)


def action_entropy(policy_logits):
    """
    Compute entropy of action distribution H[pi(a|o)].

    Args:
        policy_logits: (n_actions,) or (batch, n_actions) array

    Returns:
        entropy: scalar or (batch,) array in nats
    """
    logits = np.asarray(policy_logits, dtype=float)
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[np.newaxis]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    H = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return H[0] if squeeze else H


def entropy_reduction(logits_uncond, logits_cond):
    """
    Compute entropy reduction from conditioning on additional information.

    H[pi(a|o)] - H[pi(a|o,g)]

    Args:
        logits_uncond: (batch, n_actions) unconditional policy logits
        logits_cond: (batch, n_actions) conditional policy logits

    Returns:
        delta_H: (batch,) entropy reduction per timestep
    """
    H_uncond = action_entropy(logits_uncond)
    H_cond = action_entropy(logits_cond)
    return H_uncond - H_cond


def kl_divergence(p_logits, q_logits):
    """
    Compute KL divergence KL(P || Q) from policy logits.

    Args:
        p_logits: (n_actions,) logits for distribution P
        q_logits: (n_actions,) logits for distribution Q

    Returns:
        kl: scalar KL divergence (nats), >= 0
    """
    p_logits = np.asarray(p_logits, dtype=float)
    q_logits = np.asarray(q_logits, dtype=float)

    p = np.exp(p_logits - p_logits.max())
    p /= p.sum()
    q = np.exp(q_logits - q_logits.max())
    q /= q.sum()

    # KL(P||Q) = sum p * log(p/q)
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))
    return max(0.0, kl)
