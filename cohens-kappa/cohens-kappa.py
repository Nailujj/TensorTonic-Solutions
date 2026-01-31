import numpy as np

def cohens_kappa(rater1, rater2):
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)

    if rater1.size == 0 or rater2.size == 0:
        return 0.0
    if rater1.shape[0] != rater2.shape[0]:
        raise ValueError("rater1 and rater2 must have the same length")

    p0 = np.mean(rater1 == rater2)

    labels = np.union1d(rater1, rater2)
    pe = 0.0
    for k in labels:
        p1 = np.mean(rater1 == k)
        p2 = np.mean(rater2 == k)
        pe += p1 * p2

    denom = 1.0 - pe
    if denom == 0.0:
        return 1.0 if p0 == 1.0 else 0.0

    return (p0 - pe) / denom
