import numpy as np
from sklearn.cluster import KMeans


def instantaneous_frequency(iq, fs):
    phase = np.unwrap(np.angle(iq))
    return np.diff(phase) * fs / (2.0 * np.pi)


def extract_fsk_features(iq, fs):

    freq = instantaneous_frequency(iq, fs)

    lo, hi = np.percentile(freq, [2, 98])

    freq = freq[
        (freq >= lo) &
        (freq <= hi)
    ]

    freq = freq[::5]

    if len(freq) < 100:
        return np.zeros(18, dtype=float)

    q = np.percentile(
        freq,
        [5, 10, 25, 50, 75, 90, 95]
    )

    mean = np.mean(freq)
    std = np.std(freq)

    hist, edges = np.histogram(
        freq,
        bins=40
    )

    p = hist.astype(float)

    if p.sum() > 0:
        p /= p.sum()

    nz = p[p > 0]

    entropy = (
        -np.sum(nz * np.log2(nz))
        if len(nz)
        else 0.0
    )

    peak_bins = 0

    if len(hist) >= 3:

        for i in range(1, len(hist) - 1):

            if (
                hist[i] > hist[i - 1]
                and hist[i] >= hist[i + 1]
                and hist[i] > 0.10 * np.max(hist)
            ):
                peak_bins += 1

    X = freq.reshape(-1, 1)

    km2 = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=5
    )

    lab2 = km2.fit_predict(X)

    centers2 = np.sort(
        km2.cluster_centers_.ravel()
    )

    sep2 = (
        centers2[1] - centers2[0]
    )

    km4 = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=5
    )

    lab4 = km4.fit_predict(X)

    centers4 = np.sort(
        km4.cluster_centers_.ravel()
    )

    gaps4 = np.diff(centers4)

    mean_gap4 = np.mean(gaps4)

    gap_cv4 = (
        np.std(gaps4) / mean_gap4
        if mean_gap4 > 0
        else 0.0
    )

    def cluster_spread(labels, centers):

        total = 0.0

        for idx, center in enumerate(centers):

            pts = freq[labels == idx]

            if len(pts) == 0:
                continue

            total += (
                len(pts) *
                np.std(pts)
            )

        return total / len(freq)

    spread2 = cluster_spread(
        lab2,
        centers2
    )

    spread4 = cluster_spread(
        lab4,
        centers4
    )

    counts4 = np.bincount(
        lab4,
        minlength=4
    )

    props4 = (
        counts4 /
        max(np.sum(counts4), 1)
    )

    return np.array(
        [
            mean,
            std,
            q[0],
            q[1],
            q[2],
            q[3],
            q[4],
            q[5],
            q[6],
            entropy,
            peak_bins,
            sep2,
            mean_gap4,
            gap_cv4,
            spread2,
            spread4,
            props4[0],
            props4[-1],
        ],
        dtype=float
    )