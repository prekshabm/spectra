import numpy as np
from sklearn.mixture import GaussianMixture


def estimate_fsk_states(signal, fs):
    """
    Estimate whether an FSK signal contains approximately
    2 or 4 stable frequency states.

    Returns:
        {
            "state_count": 2 or 4,
            "score_2": ...,
            "score_4": ...,
            "confidence": ...
        }
    """

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    x = x[np.isfinite(x)]

    if len(x) < 256:
        return {
            "state_count": 2,
            "score_2": 0.0,
            "score_4": 0.0,
            "confidence": 0.0,
        }

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    power = np.mean(
        np.abs(x) ** 2
    )

    x = x / (
        np.sqrt(power)
        + 1e-12
    )

    # --------------------------------------------------------
    # Instantaneous frequency
    # --------------------------------------------------------

    phase_diff = np.angle(
        x[1:]
        * np.conj(x[:-1])
    )

    inst_freq = (
        phase_diff
        * fs
        / (2.0 * np.pi)
    )

    # --------------------------------------------------------
    # Remove extreme outliers caused by noise/transitions
    # --------------------------------------------------------

    lo = np.percentile(
        inst_freq,
        5
    )

    hi = np.percentile(
        inst_freq,
        95
    )

    valid = (
        (inst_freq >= lo)
        & (inst_freq <= hi)
    )

    inst_freq = inst_freq[valid]

    if len(inst_freq) < 128:
        return {
            "state_count": 2,
            "score_2": 0.0,
            "score_4": 0.0,
            "confidence": 0.0,
        }

    # --------------------------------------------------------
    # Smooth frequency trajectory
    # --------------------------------------------------------

    window = 9

    kernel = np.ones(
        window
    ) / window

    smooth_freq = np.convolve(
        inst_freq,
        kernel,
        mode="same"
    )

    # Downsample to reduce symbol-transition noise
    step = max(
        1,
        len(smooth_freq) // 8000
    )

    samples = smooth_freq[::step]

    samples = samples.reshape(
        -1,
        1
    )

    # --------------------------------------------------------
    # Fit 2-state and 4-state Gaussian mixtures
    # --------------------------------------------------------

    models = {}

    for k in (2, 4):

        try:

            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=5,
                reg_covar=1e-6,
            )

            gm.fit(
                samples
            )

            bic = gm.bic(
                samples
            )

            means = np.sort(
                gm.means_.ravel()
            )

            cov = np.maximum(
                gm.covariances_.ravel(),
                1e-12
            )

            separation = np.diff(
                means
            )

            if len(separation) == 0:
                min_sep = 0.0
            else:
                min_sep = float(
                    np.min(
                        np.abs(
                            separation
                        )
                    )
                )

            spread = float(
                np.sqrt(
                    np.max(cov)
                )
            )

            separation_ratio = (
                min_sep
                / (
                    spread
                    + 1e-9
                )
            )

            models[k] = {
                "bic": float(bic),
                "means": means,
                "separation_ratio":
                    float(separation_ratio),
            }

        except Exception:

            models[k] = {
                "bic": np.inf,
                "means": np.array([]),
                "separation_ratio": 0.0,
            }

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    bic2 = models[2]["bic"]
    bic4 = models[4]["bic"]

    sep2 = models[2]["separation_ratio"]
    sep4 = models[4]["separation_ratio"]

    # Lower BIC = better model fit.
    #
    # We also require 4-state separation to be meaningfully
    # stronger than noise before selecting four states.

    delta_bic = bic2 - bic4

    four_state_evidence = (
        delta_bic > 20.0
        and sep4 > 1.5
    )

    two_state_evidence = (
        delta_bic < -10.0
        and sep2 > 1.2
    )

    if four_state_evidence:

        state_count = 4

        score_4 = (
            delta_bic
            * min(
                sep4 / 3.0,
                1.0
            )
        )

        score_2 = max(
            0.0,
            20.0 - delta_bic
        )

    elif two_state_evidence:

        state_count = 2

        score_2 = (
            -delta_bic
            * min(
                sep2 / 3.0,
                1.0
            )
        )

        score_4 = max(
            0.0,
            10.0 + delta_bic
        )

    else:

        # Ambiguous case: use the better BIC model
        state_count = (
            4
            if bic4 < bic2
            else 2
        )

        score_2 = max(
            0.0,
            -delta_bic
        )

        score_4 = max(
            0.0,
            delta_bic
        )

    total = (
        score_2
        + score_4
        + 1e-12
    )

    p2 = score_2 / total
    p4 = score_4 / total

    confidence = (
        max(
            p2,
            p4
        )
    )

    return {
        "state_count": state_count,
        "score_2": float(p2),
        "score_4": float(p4),
        "confidence": float(confidence),
        "bic_2": float(bic2),
        "bic_4": float(bic4),
        "separation_2": float(sep2),
        "separation_4": float(sep4),
    }