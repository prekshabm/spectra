import numpy as np


# ============================================================
# SPECTRA — FEC / ERROR-CONTROL ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze a recovered bitstream for statistical evidence
#   consistent with error-control coding.
#
# Important:
#   This is an evidence detector, NOT a decoder.
#
#   It reports:
#
#       NONE
#       POSSIBLE
#       LIKELY
#
#   It does not claim to identify an exact FEC code with
#   certainty.
# ============================================================


# ============================================================
# INPUT CLEANING
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


# ============================================================
# BASIC STATISTICS
# ============================================================

def _bit_balance(bits):

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


# ============================================================
# SINGLE-PARITY / CHECK-BIT DETECTION
# ============================================================

def _parity_evidence(bits):
    """
    Look for a repeated parity/check-bit organization.

    The final bit of each group is treated as a candidate
    parity bit and compared against XOR parity of the
    preceding bits.

    A random stream should be close to 50% agreement.
    """

    if len(bits) < 32:

        return {
            "best_group_size": None,
            "parity_consistency": 0.0,
            "raw_consistency": 0.0,
        }

    best_size = None
    best_evidence = 0.0
    best_consistency = 0.0

    for group_size in (
        8,
        16,
        32,
        64,
        128,
    ):

        n_groups = (
            len(bits)
            //
            group_size
        )

        if n_groups < 4:
            continue

        usable = (
            n_groups
            *
            group_size
        )

        x = bits[
            :usable
        ].reshape(
            n_groups,
            group_size
        )

        data = x[:, :-1]
        parity = x[:, -1]

        predicted = (
            np.sum(
                data,
                axis=1
            )
            %
            2
        )

        consistency = float(
            np.mean(
                predicted == parity
            )
        )

        # Evidence is distance from random 50%.
        evidence = (
            abs(
                consistency
                -
                0.50
            )
            * 2.0
        )

        if evidence > best_evidence:

            best_evidence = evidence
            best_size = group_size
            best_consistency = consistency

    return {
        "best_group_size": best_size,
        "parity_consistency": round(
            best_evidence,
            4
        ),
        "raw_consistency": round(
            best_consistency,
            4
        ),
    }


# ============================================================
# TRUE REPETITION DETECTION
# ============================================================

def _repetition_evidence(bits):
    """
    Detect repetition coding.

    Supports:

    1. Bit repetition:
           0 -> 000
           1 -> 111

    2. Whole-block repetition:
           payload | payload | payload

    Returns all fields expected by analyze_fec().
    """

    n = len(bits)

    if n < 12:
        return {
            "best_repeat": None,
            "repetition_score": 0.0,
            "copy_agreement": 0.0,
        }

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    best_repeat = None
    best_score = 0.0
    best_copy_agreement = 0.0

    # ========================================================
    # BIT-LEVEL REPETITION
    # ========================================================

    for repeat in (2, 3, 4):

        usable = (
            n // repeat
        ) * repeat

        if usable < 12:
            continue

        x = bits[
            :usable
        ].reshape(
            -1,
            repeat
        )

        identical = np.all(
            x == x[:, [0]],
            axis=1
        )

        score = float(
            np.mean(
                identical
            )
        )

        if score > best_score:

            best_score = score
            best_repeat = repeat
            best_copy_agreement = score

    # ========================================================
    # WHOLE-BLOCK REPETITION
    # ========================================================

    for repeat in (2, 3, 4):

        block_size = n // repeat

        if block_size < 16:
            continue

        usable = (
            block_size
            * repeat
        )

        x = bits[
            :usable
        ].reshape(
            repeat,
            block_size
        )

        agreements = []

        for k in range(
            1,
            repeat
        ):

            agreement = float(
                np.mean(
                    x[0] == x[k]
                )
            )

            agreements.append(
                agreement
            )

        if agreements:

            score = float(
                np.mean(
                    agreements
                )
            )

            if score > best_score:

                best_score = score
                best_repeat = repeat
                best_copy_agreement = score

    # ========================================================
    # RANDOM-DATA PROTECTION
    # ========================================================

    if best_score < 0.70:

        return {
            "best_repeat": None,
            "repetition_score": 0.0,
            "copy_agreement": round(
                best_score,
                4
            ),
        }

    # ========================================================
    # NORMALIZED EVIDENCE
    # ========================================================

    repetition_score = float(
        np.clip(
            (best_score - 0.50) / 0.50,
            0.0,
            1.0
        )
    )

    return {
        "best_repeat": int(
            best_repeat
        ),
        "repetition_score": round(
            repetition_score,
            4
        ),
        "copy_agreement": round(
            best_copy_agreement,
            4
        ),
    }


# ============================================================
# HAMMING (7,4) DETECTION
# ============================================================

def _hamming74_evidence(bits):
    """
    Test whether consecutive 7-bit codewords are consistent
    with the standard binary Hamming (7,4) parity structure.

    Codeword positions:

        1 2 3 4 5 6 7

    where positions 1, 2 and 4 are parity positions.

    Syndrome:

        s1 = b1 XOR b3 XOR b5 XOR b7
        s2 = b2 XOR b3 XOR b6 XOR b7
        s4 = b4 XOR b5 XOR b6 XOR b7

    For a valid codeword, all syndrome bits are zero.
    """

    if len(bits) < 70:

        return {
            "code": "Hamming(7,4)",
            "codeword_size": 7,
            "valid_codeword_rate": 0.0,
            "syndrome_evidence": 0.0,
            "codeword_count": 0,
        }

    usable = (
        len(bits)
        //
        7
    ) * 7

    x = bits[
        :usable
    ].reshape(
        -1,
        7
    )

    s1 = (
        x[:, 0]
        ^
        x[:, 2]
        ^
        x[:, 4]
        ^
        x[:, 6]
    )

    s2 = (
        x[:, 1]
        ^
        x[:, 2]
        ^
        x[:, 5]
        ^
        x[:, 6]
    )

    s4 = (
        x[:, 3]
        ^
        x[:, 4]
        ^
        x[:, 5]
        ^
        x[:, 6]
    )

    valid = (
        (s1 == 0)
        &
        (s2 == 0)
        &
        (s4 == 0)
    )

    valid_rate = float(
        np.mean(valid)
    )

    # Random 7-bit words have approximately 1/8 valid
    # syndrome combinations.
    baseline = 1.0 / 8.0

    if valid_rate <= baseline:
        evidence = 0.0
    else:
        evidence = (
            valid_rate
            -
            baseline
        ) / (
            1.0
            -
            baseline
        )

    evidence = float(
        np.clip(
            evidence,
            0.0,
            1.0
        )
    )

    return {
        "code": "Hamming(7,4)",
        "codeword_size": 7,
        "valid_codeword_rate": round(
            valid_rate,
            4
        ),
        "syndrome_evidence": round(
            evidence,
            4
        ),
        "codeword_count": int(
            len(x)
        ),
    }


# ============================================================
# BLOCK CORRELATION
# ============================================================

def _block_correlation(bits):

    if len(bits) < 64:

        return {
            "best_block_size": None,
            "best_correlation": 0.0,
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

    best_size = None
    best_corr = 0.0

    for block_size in (
        8,
        16,
        32,
        64,
        128,
        256,
    ):

        n_blocks = (
            len(x)
            //
            block_size
        )

        if n_blocks < 3:
            continue

        usable = (
            x[
                :n_blocks * block_size
            ]
            .reshape(
                n_blocks,
                block_size
            )
        )

        correlations = []

        for k in range(
            n_blocks - 1
        ):

            a = usable[k]
            b = usable[k + 1]

            a = (
                a
                -
                np.mean(a)
            )

            b = (
                b
                -
                np.mean(b)
            )

            den = np.sqrt(
                np.dot(a, a)
                *
                np.dot(b, b)
            )

            if den <= 1e-12:
                continue

            correlations.append(
                abs(
                    float(
                        np.dot(a, b)
                        /
                        den
                    )
                )
            )

        if not correlations:
            continue

        score = float(
            np.mean(
                correlations
            )
        )

        if score > best_corr:

            best_corr = score
            best_size = block_size

    return {
        "best_block_size": best_size,
        "best_correlation": round(
            best_corr,
            4
        ),
    }


# ============================================================
# CODE-RATE / LENGTH EVIDENCE
# ============================================================

def _length_evidence(bits):

    n = len(bits)

    candidates = []

    rates = {
        "1/2": 2,
        "2/3": 3,
        "3/4": 4,
        "4/5": 5,
        "7/8": 8,
    }

    for name, denominator in rates.items():

        if (
            n >= denominator
            and
            n % denominator == 0
        ):

            candidates.append(
                name
            )

    return {
        "length": int(n),
        "rate_candidates": candidates,
    }


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def _classify_fec(
    bits,
    parity,
    repetition,
    hamming,
    block,
):

    score = 0.0
    reasons = []

    parity_score = float(
        parity.get(
            "parity_consistency",
            0.0
        )
    )

    repetition_score = float(
        repetition.get(
            "repetition_score",
            0.0
        )
    )

    hamming_score = float(
        hamming.get(
            "syndrome_evidence",
            0.0
        )
    )

    block_corr = float(
        block.get(
            "best_correlation",
            0.0
        )
    )

    # ========================================================
    # STRONG STRUCTURAL EVIDENCE
    # ========================================================
    #
    # These are highly specific patterns and therefore carry
    # more weight than generic statistical correlation.
    # ========================================================

    # --------------------------------------------------------
    # REPETITION
    # --------------------------------------------------------

    if repetition_score >= 0.90:

        score += 0.70

        reasons.append(
            "strong repeated-payload structure"
        )

    elif repetition_score >= 0.70:

        score += 0.45

        reasons.append(
            "moderate repeated-bit structure"
        )

    elif repetition_score >= 0.50:

        score += 0.25

        reasons.append(
            "weak repeated-bit structure"
        )

    # --------------------------------------------------------
    # HAMMING
    # --------------------------------------------------------

    if hamming_score >= 0.80:

        score += 0.70

        reasons.append(
            "strong Hamming(7,4) syndrome consistency"
        )

    elif hamming_score >= 0.50:

        score += 0.45

        reasons.append(
            "moderate Hamming(7,4) syndrome consistency"
        )

    elif hamming_score >= 0.30:

        score += 0.25

        reasons.append(
            "weak Hamming(7,4) syndrome consistency"
        )

    # --------------------------------------------------------
    # PARITY
    # --------------------------------------------------------

    if parity_score >= 0.90:

        score += 0.45

        reasons.append(
            "strong parity/check-bit consistency"
        )

    elif parity_score >= 0.75:

        score += 0.30

        reasons.append(
            "moderate parity/check-bit consistency"
        )

    elif parity_score >= 0.60:

        score += 0.15

        reasons.append(
            "weak parity/check-bit consistency"
        )

    # --------------------------------------------------------
    # BLOCK CORRELATION
    #
    # Block correlation by itself is not enough to claim FEC.
    # It is supporting evidence only.
    # --------------------------------------------------------

    if block_corr >= 0.90:

        score += 0.15

        reasons.append(
            "strong block correlation"
        )

    elif block_corr >= 0.70:

        score += 0.10

        reasons.append(
            "moderate block correlation"
        )

    # ========================================================
    # CAP SCORE
    # ========================================================

    score = float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if score >= 0.70:

        classification = "LIKELY"

    elif score >= 0.30:

        classification = "POSSIBLE"

    else:

        classification = "NONE"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = float(
        score
    )

    return (
        classification,
        confidence,
        reasons
    )


# ============================================================
# PUBLIC API
# ============================================================

def analyze_fec(bits):
    """
    Analyze a recovered bitstream for possible FEC.

    Parameters
    ----------
    bits:
        String such as "010101..."
        or an array/list containing 0 and 1.

    Returns
    -------
    dict
        FEC-analysis results.
    """

    bits = _clean_bits(
        bits
    )

    balance = _bit_balance(
        bits
    )

    parity = _parity_evidence(
        bits
    )

    repetition = _repetition_evidence(
        bits
    )

    hamming = _hamming74_evidence(
        bits
    )

    block = _block_correlation(
        bits
    )

    length = _length_evidence(
        bits
    )

    (
        classification,
        score,
        reasons
    ) = _classify_fec(
        bits,
        parity,
        repetition,
        hamming,
        block,
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

        "parity_group_size": parity[
            "best_group_size"
        ],

        "parity_evidence": parity[
            "parity_consistency"
        ],

        "best_repeat": repetition[
            "best_repeat"
        ],

        "repetition_evidence": repetition[
            "repetition_score"
        ],

        "copy_agreement": repetition[
            "copy_agreement"
        ],

        "hamming_code": hamming[
            "code"
        ],

        "hamming_codeword_size": hamming[
            "codeword_size"
        ],

        "hamming_valid_codeword_rate": hamming[
            "valid_codeword_rate"
        ],

        "hamming_syndrome_evidence": hamming[
            "syndrome_evidence"
        ],

        "hamming_codeword_count": hamming[
            "codeword_count"
        ],

        "best_block_size": block[
            "best_block_size"
        ],

        "block_correlation": block[
            "best_correlation"
        ],

        "code_rate_candidates": length[
            "rate_candidates"
        ],

        "fec_evidence": classification,

        "confidence": round(
            score * 100.0,
            2
        ),

        "evidence_reasons": reasons,
    }