"""Baseline representation statistics for E5: linear CKA, linear-probe
accuracy, and the neural-collapse within/between scatter ratio (NC1).

Total mixup (Wagner et al. 2024) is intentionally not re-implemented here;
the comparison uses their reference implementation, vendored separately, so
the baseline is theirs and not our re-reading of it.  See cluster/README.
"""

import numpy as np


def linear_cka(X, Y):
    """Linear CKA between two feature matrices on the same examples."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy + 1e-12))


def linear_probe_acc(X, y, seed=0):
    """5-fold CV accuracy of a multinomial logistic probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, n_jobs=1)))


def nc1(X, y):
    """Neural collapse NC1: tr(S_W S_B^+) / k (Papyan et al. 2020)."""
    classes = np.unique(y)
    mu_g = X.mean(0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        D = Xc - mu
        Sw += D.T @ D / len(Xc)
        d = (mu - mu_g)[:, None]
        Sb += d @ d.T
    Sw /= len(classes)
    Sb /= len(classes)
    return float(np.trace(Sw @ np.linalg.pinv(Sb)) / len(classes))
