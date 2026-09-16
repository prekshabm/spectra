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