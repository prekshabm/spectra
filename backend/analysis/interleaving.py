import numpy as np


# ============================================================
# SPECTRA — INTERLEAVING ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze an estimated bitstream for statistical evidence
#   of possible interleaving.
#
# Important:
#   Statistical analysis alone cannot prove that an arbitrary
#   bitstream was interleaved. Therefore this module reports:
#
#       NONE
#       POSSIBLE
#       LIKELY
#
#   rather than making an absolute detection claim.
# ============================================================


def _clean_bits(bits):
    """Convert a bitstream into a clean uint8 array."""

    if isinstance(bits, str):

        bits = [
            int(b)
            for b in bits
            if b in ("0", "1")
        ]

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    bits = bits[
        (bits == 0)
        |
        (bits == 1)
    ]

    return bits


def _bit_balance(bits):
    """Calculate zero/one percentages."""

    if len(bits) == 0:

        return {
            "zero_percentage": 0.0,
            "one_percentage": 0.0,
        }

    zero = float(
        np.mean(bits == 0)
        * 100.0
    )

    one = float(
        np.mean(bits == 1)
        * 100.0
    )

    return {
        "zero_percentage": round(
            zero,
            2
        ),
        "one_percentage": round(
            one,
            2
        ),
    }


def _transition_rate(bits):
    """
    Calculate the fraction of adjacent bits that change.

    Random-like streams tend toward approximately 50%.
    """

    if len(bits) < 2:
        return 0.0

    transitions = np.mean(
        bits[1:] != bits[:-1]
    )

    return float(
        transitions
    )


def _run_statistics(bits):
    """
    Analyze consecutive runs of identical bits.
    """

    if len(bits) == 0:

        return {
            "mean_run_length": 0.0,
            "max_run_length": 0,
            "run_count": 0,
        }

    changes = np.where(
        bits[1:] != bits[:-1]
    )[0] + 1

    boundaries = np.concatenate(
        (
            [0],
            changes,
            [len(bits)]
        )
    )

    lengths = np.diff(
        boundaries
    )

    return {
        "mean_run_length": round(
            float(np.mean(lengths)),
            3
        ),
        "max_run_length": int(
            np.max(lengths)
        ),
        "run_count": int(
            len(lengths)
        ),
    }


def _autocorrelation(bits, max_lag=128):
    """
    Calculate normalized lag correlation.

    The bit sequence is mapped from:

        0 -> -1
        1 -> +1
    """

    if len(bits) < 8:

        return {
            "strongest_lag": None,
            "strongest_correlation": 0.0,
            "periodicity_score": 0.0,
        }

    x = (
        bits.astype(
            np.float64
        )
        *
        2.0
        -
        1.0
    )

    x = x - np.mean(x)

    denominator = np.dot(
        x,
        x
    )

    if denominator <= 1e-12:

        return {
            "strongest_lag": None,
            "strongest_correlation": 0.0,
            "periodicity_score": 0.0,
        }

    max_lag = min(
        int(max_lag),
        len(x) // 2
    )

    correlations = []

    for lag in range(
        1,
        max_lag + 1
    ):

        a = x[:-lag]
        b = x[lag:]

        den = np.sqrt(
            np.dot(a, a)
            *
            np.dot(b, b)
        )

        if den <= 1e-12:

            corr = 0.0

        else:

            corr = (
                np.dot(a, b)
                /
                den
            )

        correlations.append(
            float(corr)
        )

    if not correlations:

        return {
            "strongest_lag": None,
            "strongest_correlation": 0.0,
            "periodicity_score": 0.0,
        }

    correlations = np.asarray(
        correlations
    )

    idx = int(
        np.argmax(
            np.abs(
                correlations
            )
        )
    )

    strongest = float(
        correlations[idx]
    )

    return {
        "strongest_lag": int(
            idx + 1
        ),
        "strongest_correlation": round(
            strongest,
            4
        ),
        "periodicity_score": round(
            float(
                abs(strongest)
            ),
            4
        ),
        "lags": list(
            range(
                1,
                len(correlations) + 1
            )
        ),
        "correlations": [
            round(
                float(value),
                4
            )
            for value in correlations
        ],
    }


def _block_statistics(bits):
    """
    Compare statistics between consecutive blocks.

    Interleaving can sometimes alter local structure,
    but block statistics alone are not proof of interleaving.
    """

    if len(bits) < 32:
        return {
            "block_size": 0,
            "blocks": 0,
            "balance_variation": 0.0,
        }

    block_size = max(
        16,
        min(
            256,
            len(bits) // 8
        )
    )

    blocks = []

    for start in range(
        0,
        len(bits),
        block_size
    ):

        block = bits[
            start:
            start + block_size
        ]

        if len(block) >= 8:

            blocks.append(
                float(
                    np.mean(block)
                )
            )

    if len(blocks) < 2:

        variation = 0.0

    else:

        variation = float(
            np.std(blocks)
        )

    return {
        "block_size": int(
            block_size
        ),
        "blocks": int(
            len(blocks)
        ),
        "balance_variation": round(
            variation,
            4
        ),
    }

def _classify_evidence(
    bits,
    transition_rate,
    autocorrelation,
    block_statistics,
    
):
    """
    Produce conservative interleaving evidence.

    This is a heuristic detector. Statistical and structural
    evidence cannot prove interleaving.
    """

    score = 0.0

    # --------------------------------------------------------
    # Structural interleaving evidence
    # --------------------------------------------------------

    

    # --------------------------------------------------------
    # Transition structure
    # --------------------------------------------------------

    if (
        transition_rate < 0.20
        or
        transition_rate > 0.80
    ):
        score += 0.25

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

    corr = abs(
        autocorrelation.get(
            "strongest_correlation",
            0.0
        )
    )

    if corr > 0.50:
        score += 0.20

    # --------------------------------------------------------
    # Block imbalance
    # --------------------------------------------------------

    variation = block_statistics.get(
        "balance_variation",
        0.0
    )

    if variation > 0.15:
        score += 0.20

    # --------------------------------------------------------
    # Short-stream penalty
    # --------------------------------------------------------

    if len(bits) < 128:
        score *= 0.50

    score = float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

    if score >= 0.60:
        classification = "LIKELY"

    elif score >= 0.30:
        classification = "POSSIBLE"

    else:
        classification = "NONE"

    return classification, score

def _interleaving_structure(bits):
    """
    Look for structural evidence associated with common
    deterministic interleavers.

    This is a heuristic detector. It does not prove that
    interleaving is present.

    Returns:
        dict containing structural scores and best candidate.
    """

    n = len(bits)

    if n < 64:
        return {
            "block_score": 0.0,
            "matrix_score": 0.0,
            "stride_score": 0.0,
            "best_score": 0.0,
            "best_type": None,
        }

    x = bits.astype(np.float64)

    # --------------------------------------------------------
    # Compare adjacent chunks.
    #
    # A deterministic block interleaver can preserve related
    # statistics between regularly spaced chunks.
    # --------------------------------------------------------

    block_scores = []

    for block_size in (
        16,
        32,
        64,
        128,
        256,
    ):

        if block_size >= n:
            continue

        blocks = []

        for start in range(
            0,
            n - block_size + 1,
            block_size
        ):

            block = x[
                start:
                start + block_size
            ]

            if len(block) == block_size:
                blocks.append(block)

        if len(blocks) < 2:
            continue

        # Compare block-level transition statistics.
        local_rates = []

        for block in blocks:

            if len(block) < 2:
                continue

            local_rates.append(
                np.mean(
                    block[1:] != block[:-1]
                )
            )

        if len(local_rates) >= 2:

            variation = float(
                np.std(local_rates)
            )

            # Normalize to a useful heuristic range.
            score = float(
                np.clip(
                    variation / 0.20,
                    0.0,
                    1.0
                )
            )

            block_scores.append(score)

    block_score = (
        max(block_scores)
        if block_scores
        else 0.0
    )

    # --------------------------------------------------------
    # Matrix/stride structure.
    #
    # Test regularly spaced subsequences. Interleaving often
    # converts sequential structure into column/stride
    # structure.
    # --------------------------------------------------------

    stride_scores = []

    for stride in (
        2,
        4,
        8,
        16,
        32,
    ):

        if stride >= n:
            continue

        correlations = []

        for offset in range(stride):

            seq = x[offset::stride]

            if len(seq) < 16:
                continue

            seq = (
                seq
                - np.mean(seq)
            )

            den = np.dot(
                seq,
                seq
            )

            if den <= 1e-12:
                continue

            # Adjacent samples inside the extracted stride
            # sequence.
            a = seq[:-1]
            b = seq[1:]

            corr_den = np.sqrt(
                np.dot(a, a)
                *
                np.dot(b, b)
            )

            if corr_den > 1e-12:

                correlations.append(
                    abs(
                        float(
                            np.dot(a, b)
                            /
                            corr_den
                        )
                    )
                )

        if correlations:

            stride_scores.append(
                max(correlations)
            )

    stride_score = (
        max(stride_scores)
        if stride_scores
        else 0.0
    )

    # --------------------------------------------------------
    # Matrix candidates.
    #
    # Reshape into approximately square matrices and compare
    # row/column transition structure.
    # --------------------------------------------------------

    matrix_scores = []

    for cols in (
        8,
        16,
        32,
        64,
    ):

        rows = n // cols

        if rows < 4:
            continue

        usable = rows * cols

        matrix = x[
            :usable
        ].reshape(
            rows,
            cols
        )

        row_rates = np.mean(
            matrix[:, 1:] != matrix[:, :-1],
            axis=1
        )

        col_rates = np.mean(
            matrix[1:, :] != matrix[:-1, :],
            axis=0
        )

        if len(row_rates) and len(col_rates):

            row_mean = float(
                np.mean(row_rates)
            )

            col_mean = float(
                np.mean(col_rates)
            )

            difference = abs(
                row_mean
                -
                col_mean
            )

            matrix_scores.append(
                float(
                    np.clip(
                        difference / 0.25,
                        0.0,
                        1.0
                    )
                )
            )

    matrix_score = (
        max(matrix_scores)
        if matrix_scores
        else 0.0
    )

    candidates = {
        "BLOCK": block_score,
        "MATRIX": matrix_score,
        "STRIDE": stride_score,
    }

    best_type = max(
        candidates,
        key=candidates.get
    )

    best_score = float(
        candidates[best_type]
    )

    # Avoid reporting tiny random fluctuations as structure.
    if best_score < 0.30:
        best_type = None

    return {
        "block_score": round(
            block_score,
            4
        ),
        "matrix_score": round(
            matrix_score,
            4
        ),
        "stride_score": round(
            stride_score,
            4
        ),
        "best_score": round(
            best_score,
            4
        ),
        "best_type": best_type,
    }

# ============================================================
# PUBLIC API
# ============================================================

def analyze_interleaving(bits):
    """
    Analyze an estimated bitstream for possible interleaving.

    Parameters
    ----------
    bits:
        String such as "010101..." or an array/list
        containing 0 and 1.

    Returns
    -------
    dict
        Interleaving-analysis results.
    """

    bits = _clean_bits(
        bits
    )

    balance = _bit_balance(
        bits
    )

    transition = _transition_rate(
        bits
    )

    runs = _run_statistics(
        bits
    )

    autocorrelation = _autocorrelation(
        bits
    )

    block_statistics = _block_statistics(
        bits
    )

    block_statistics = _block_statistics(
        bits
    )

    structure = _interleaving_structure(
        bits
    )

    classification, score = (
        _classify_evidence(
            bits,
            transition,
            autocorrelation,
            block_statistics,
            
        )
    )

    return {

        "bits_analyzed": int(
            len(bits)
        ),

        "zero_percentage": balance[
            "zero_percentage"
        ],

        "one_percentage": balance[
            "one_percentage"
        ],

        "transition_rate": round(
            transition,
            4
        ),

        "transition_percentage": round(
            transition * 100.0,
            2
        ),

        "mean_run_length": runs[
            "mean_run_length"
        ],

        "max_run_length": runs[
            "max_run_length"
        ],

        "run_count": runs[
            "run_count"
        ],

        "strongest_lag": (
            autocorrelation[
                "strongest_lag"
            ]
        ),

        "strongest_correlation": (
            autocorrelation[
                "strongest_correlation"
            ]
        ),

        "periodicity_score": (
            autocorrelation[
                "periodicity_score"
            ]
        ),

        "block_size": (
            block_statistics[
                "block_size"
            ]
        ),

        "blocks_analyzed": (
            block_statistics[
                "blocks"
            ]
        ),

        "block_balance_variation": (
            block_statistics[
                "balance_variation"
            ]
        ),

        "interleaving_evidence": classification,

        "confidence": round(
            score * 100.0,
            2
        ),

        "note": (
            "Statistical evidence only. "
            "Interleaving cannot be proven from "
            "an arbitrary bitstream without "
            "framing, coding, or known-reference "
            "information."
        ),
    }


class InterleavingAnalyzer:

    def analyze(self, bits):

        return analyze_interleaving(
            bits
        )


# ============================================================
# AUTOMATIC FOUR-WAY INTERLEAVING DETECTION + DEINTERLEAVING
# ============================================================

def _repeat_structure_score(bits):
    """
    Estimate repeated-frame / repeated-block structure.

    This is deliberately conservative: it is only a recoverability
    signal used to compare deinterleaver candidates. It does not
    identify a protocol or prove an interleaving scheme.
    """

    n = len(bits)

    if n < 64:
        return 0.0

    best = 0.0

    for size in (16, 32, 64, 128, 256):
        blocks = n // size

        if blocks < 3:
            continue

        x = bits[:blocks * size].reshape(blocks, size)

        reference = x[0]
        agreement = np.mean(x[1:] == reference)

        if agreement > best:
            best = float(agreement)

    # Random binary streams tend toward ~0.5 agreement.
    # Map 0.5..1.0 into 0..1 conservatively.
    return float(
        np.clip(
            (best - 0.50) / 0.50,
            0.0,
            1.0
        )
    )


def _candidate_recoverability(bits):
    """
    Score how much recoverable sequential structure is present.

    Higher is better. This is used only to rank supported
    deinterleaver candidates against the original stream.
    """

    bits = _clean_bits(bits)

    if len(bits) < 64:
        return {
            "quality": 0.0,
            "periodicity": 0.0,
            "repeat": 0.0,
            "interleaving_penalty": 0.0
        }

    autocorrelation = _autocorrelation(
        bits,
        max_lag=min(256, len(bits) // 2)
    )

    correlations = autocorrelation.get(
        "correlations",
        []
    )

    # Ignore the first few lags because those mostly reflect
    # local bit transitions rather than frame/block structure.
    useful = correlations[3:]

    periodicity = (
        float(np.max(np.abs(useful)))
        if useful
        else 0.0
    )

    repeat = _repeat_structure_score(bits)

    block_statistics = _block_statistics(bits)

    transition = _transition_rate(bits)

    _, interleaving_score = _classify_evidence(
        bits,
        transition,
        autocorrelation,
        block_statistics
    )

    # More recoverable structure + less unexplained interleaving
    # evidence = better candidate.
    quality = (
        0.55 * periodicity
        +
        0.30 * repeat
        +
        0.15 * (1.0 - interleaving_score)
    )

    return {
        "quality": round(
            float(np.clip(quality, 0.0, 1.0)),
            4
        ),
        "periodicity": round(
            float(periodicity),
            4
        ),
        "repeat": round(
            float(repeat),
            4
        ),
        "interleaving_penalty": round(
            float(interleaving_score),
            4
        )
    }


def _run_supported_deinterleaver(
    bits,
    mode,
    rows=8,
    branches=4,
    delay=3,
    seed=42
):
    """
    Apply one of SPECTRA's four supported deterministic
    deinterleavers.

    Returns a uint8 array or None on failure.
    """

    from backend.analysis.interleavers import (
        block_deinterleave,
        convolutional_deinterleave,
        diagonal_deinterleave,
        pseudo_random_deinterleave
    )

    try:

        if mode == "BLOCK":
            return np.asarray(
                block_deinterleave(
                    bits,
                    rows=int(rows)
                ),
                dtype=np.uint8
            )

        if mode == "CONVOLUTIONAL":
            return np.asarray(
                convolutional_deinterleave(
                    bits,
                    branches=int(branches),
                    delay=int(delay)
                ),
                dtype=np.uint8
            )

        if mode == "DIAGONAL":
            return np.asarray(
                diagonal_deinterleave(
                    bits,
                    rows=int(rows)
                ),
                dtype=np.uint8
            )

        if mode == "PSEUDO-RANDOM":
            return np.asarray(
                pseudo_random_deinterleave(
                    bits,
                    seed=int(seed)
                ),
                dtype=np.uint8
            )

    except Exception:
        return None

    return None


def auto_detect_and_deinterleave(
    bits,
    rows=8,
    branches=4,
    delay=3,
    seed=42
):
    """
    Automatically choose among the four supported interleaving
    families and apply the corresponding deinterleaver.

    Supported candidates:
        BLOCK
        CONVOLUTIONAL
        DIAGONAL
        PSEUDO-RANDOM

    This is intentionally conservative. An unknown arbitrary RF
    bitstream cannot mathematically prove its interleaver without
    additional framing/reference information. The function therefore
    requires a meaningful recoverability gain and a separation margin
    before returning VALIDATED.
    """

    clean = _clean_bits(bits)
    n = len(clean)

    if n < 128:
        return {
            "detected": False,
            "type": "UNKNOWN",
            "confidence": 0.0,
            "status": "INSUFFICIENT_DATA",
            "candidates": {},
            "deinterleaving": {
                "available": False,
                "success": False,
                "type": "NONE",
                "reason": "At least 128 bits are required for automatic four-way interleaving detection."
            }
        }

    baseline = _candidate_recoverability(clean)

    structure = _interleaving_structure(clean)

    structural_prior = {
        "BLOCK": float(
            structure.get("block_score", 0.0)
        ),
        "CONVOLUTIONAL": float(
            structure.get("stride_score", 0.0)
        ),
        "DIAGONAL": float(
            structure.get("matrix_score", 0.0)
        ),
        "PSEUDO-RANDOM": 0.0
    }

    results = {}

    for mode in (
        "BLOCK",
        "CONVOLUTIONAL",
        "DIAGONAL",
        "PSEUDO-RANDOM"
    ):

        candidate_bits = _run_supported_deinterleaver(
            clean,
            mode,
            rows=rows,
            branches=branches,
            delay=delay,
            seed=seed
        )

        if candidate_bits is None or len(candidate_bits) != n:
            results[mode] = {
                "available": False,
                "score": 0.0,
                "recoverability_gain": 0.0,
                "reason": "Candidate deinterleaver failed."
            }
            continue

        metrics = _candidate_recoverability(
            candidate_bits
        )

        gain = (
            float(metrics["quality"])
            -
            float(baseline["quality"])
        )

        # Convert gain into a conservative 0..1 score.
        gain_score = float(
            np.clip(
                (gain - 0.02) / 0.20,
                0.0,
                1.0
            )
        )

        prior = structural_prior[mode]

        # Structural prior helps the three deterministic geometric
        # families; pseudo-random relies almost entirely on the
        # observed recoverability improvement.
        if mode == "PSEUDO-RANDOM":
            candidate_score = (
                0.80 * gain_score
                +
                0.20 * metrics["repeat"]
            )
        else:
            candidate_score = (
                0.55 * gain_score
                +
                0.45 * prior
            )

        results[mode] = {
            "available": True,
            "score": round(
                float(np.clip(candidate_score, 0.0, 1.0)),
                4
            ),
            "recoverability_gain": round(
                float(gain),
                4
            ),
            "recoverability": metrics,
            "structural_prior": round(
                float(prior),
                4
            )
        }

    valid_candidates = [
        (mode, item)
        for mode, item in results.items()
        if item.get("available")
    ]

    if not valid_candidates:
        return {
            "detected": False,
            "type": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_VALID_CANDIDATE",
            "candidates": results,
            "deinterleaving": {
                "available": False,
                "success": False,
                "type": "NONE",
                "reason": "No supported deinterleaver could be evaluated."
            }
        }

    ranked = sorted(
        valid_candidates,
        key=lambda item: float(
            item[1].get("score", 0.0)
        ),
        reverse=True
    )

    best_mode, best = ranked[0]

    best_score = float(
        best.get("score", 0.0)
    )

    best_gain = float(
        best.get("recoverability_gain", 0.0)
    )

    second_score = (
        float(ranked[1][1].get("score", 0.0))
        if len(ranked) > 1
        else 0.0
    )

    margin = (
        best_score
        -
        second_score
    )

    # Conservative validation gate.
    validated = (
        best_score >= 0.50
        and
        best_gain >= 0.04
        and
        margin >= 0.08
    )

    if not validated:
        confidence = float(
            np.clip(
                max(
                    0.0,
                    best_score * 100.0
                ),
                0.0,
                100.0
            )
        )

        return {
            "detected": False,
            "type": "UNKNOWN",
            "confidence": round(
                confidence,
                2
            ),
            "status": "AMBIGUOUS",
            "best_candidate": best_mode,
            "candidates": results,
            "deinterleaving": {
                "available": False,
                "success": False,
                "type": "NONE",
                "reason": "No candidate passed the recoverability and separation validation gates."
            }
        }

    recovered = _run_supported_deinterleaver(
        clean,
        best_mode,
        rows=rows,
        branches=branches,
        delay=delay,
        seed=seed
    )

    if recovered is None:
        return {
            "detected": False,
            "type": "UNKNOWN",
            "confidence": 0.0,
            "status": "DEINTERLEAVER_FAILED",
            "best_candidate": best_mode,
            "candidates": results,
            "deinterleaving": {
                "available": False,
                "success": False,
                "type": "NONE",
                "reason": "The validated candidate could not be applied."
            }
        }

    confidence = float(
        np.clip(
            (
                0.60 * best_score
                +
                0.40 * np.clip(
                    (margin - 0.08) / 0.25,
                    0.0,
                    1.0
                )
            )
            * 100.0,
            0.0,
            100.0
        )
    )

    return {
        "detected": True,
        "type": best_mode,
        "confidence": round(
            confidence,
            2
        ),
        "status": "VALIDATED",
        "best_candidate": best_mode,
        "candidates": results,
        "baseline": baseline,
        "deinterleaving": {
            "available": True,
            "success": True,
            "type": best_mode,
            "input_bits": int(n),
            "output_bits": int(len(recovered)),
            "bitstream": "".join(
                str(int(bit))
                for bit in recovered
            )
        }
    }

