import numpy as np


# ============================================================
# BITSTREAM ANALYSIS
# ============================================================


def _bits_from_string(bitstream):
    """
    Convert a binary string into a uint8 NumPy array.
    """

    if not isinstance(bitstream, str):
        bitstream = str(bitstream)

    return np.asarray(
        [
            int(b)
            for b in bitstream
            if b in ("0", "1")
        ],
        dtype=np.uint8
    )


def _longest_run(bits, value):
    """
    Find the longest consecutive run of 0s or 1s.
    """

    if len(bits) == 0:
        return 0

    longest = 0
    current = 0

    for bit in bits:

        if int(bit) == value:
            current += 1
            longest = max(
                longest,
                current
            )
        else:
            current = 0

    return int(longest)


def _transition_rate(bits):
    """
    Fraction of adjacent bits that change.
    """

    if len(bits) < 2:
        return 0.0

    return float(
        np.mean(
            bits[1:] != bits[:-1]
        )
    )


def _autocorrelation(bits, max_lag=128):
    """
    Calculate strongest normalized binary autocorrelation
    and its corresponding lag.
    """

    if len(bits) < 16:
        return {
            "strongest_lag": None,
            "correlation": 0.0
        }

    x = (
        2.0 * bits.astype(np.float64)
    ) - 1.0

    x -= np.mean(x)

    energy = np.sum(
        x * x
    )

    if energy <= 0:
        return {
            "strongest_lag": None,
            "correlation": 0.0
        }

    best_corr = 0.0
    best_lag = None

    upper = min(
        max_lag,
        len(x) // 2
    )

    for lag in range(
        1,
        upper + 1
    ):

        a = x[:-lag]
        b = x[lag:]

        denominator = np.sqrt(
            np.sum(a * a)
            *
            np.sum(b * b)
        )

        if denominator <= 0:
            continue

        correlation = abs(
            float(
                np.sum(a * b)
                /
                denominator
            )
        )

        if correlation > best_corr:

            best_corr = correlation
            best_lag = lag

    return {
        "strongest_lag": best_lag,
        "correlation": round(
            best_corr,
            4
        )
    }


def _repeated_patterns(
    bits,
    pattern_lengths=(8, 16, 32)
):
    """
    Find repeated binary patterns.

    This is intended to expose possible
    synchronization/header structure.
    """

    results = []

    for length in pattern_lengths:

        if len(bits) < length * 2:
            continue

        patterns = {}

        for start in range(
            len(bits) - length + 1
        ):

            pattern = "".join(
                str(int(b))
                for b in bits[
                    start:
                    start + length
                ]
            )

            patterns.setdefault(
                pattern,
                []
            ).append(start)

        repeated = [
            {
                "pattern": pattern,
                "occurrences": len(positions),
                "positions": positions[:10]
            }
            for pattern, positions
            in patterns.items()
            if len(positions) >= 2
        ]

        repeated.sort(
            key=lambda item: item[
                "occurrences"
            ],
            reverse=True
        )

        results.append({
            "pattern_length": length,
            "patterns": repeated[:10]
        })

    return results


def analyze_bitstream(bitstream):
    """
    Analyze the recovered digital bitstream.

    The output contains descriptive statistics and
    structural evidence. It does not claim protocol
    or header identification without a known reference.
    """

    bits = _bits_from_string(
        bitstream
    )

    if len(bits) == 0:

        return {
            "available": False,
            "reason": (
                "No recovered binary bitstream available."
            )
        }

    zeros = int(
        np.sum(bits == 0)
    )

    ones = int(
        np.sum(bits == 1)
    )

    total = len(bits)

    autocorrelation = _autocorrelation(
        bits
    )

    return {
        "available": True,

        "bits_analyzed": total,

        "zero_count": zeros,

        "one_count": ones,

        "zero_percentage": round(
            100.0 * zeros / total,
            2
        ),

        "one_percentage": round(
            100.0 * ones / total,
            2
        ),

        "longest_zero_run": _longest_run(
            bits,
            0
        ),

        "longest_one_run": _longest_run(
            bits,
            1
        ),

        "transition_rate": round(
            _transition_rate(bits),
            4
        ),

        "strongest_lag": autocorrelation[
            "strongest_lag"
        ],

        "autocorrelation": autocorrelation[
            "correlation"
        ],

        "repeated_patterns": _repeated_patterns(
            bits
        ),

        "reference_required_for_identification": True
    }