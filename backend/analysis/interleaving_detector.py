import numpy as np

from backend.analysis.interleavers import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudo_random_deinterleave
)


# ============================================================
# VALIDATION
# ============================================================

def _validate_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) == 0:
        raise ValueError("Bitstream is empty.")

    if np.any((bits != 0) & (bits != 1)):
        raise ValueError(
            "Bitstream must contain only 0 and 1."
        )

    return bits


# ============================================================
# BIT MATCHING
# ============================================================

def _bit_match_score(reference, recovered):
    """
    Compare a recovered bitstream with a known reference.
    """

    reference = _validate_bits(reference)
    recovered = _validate_bits(recovered)

    length = min(
        len(reference),
        len(recovered)
    )

    if length == 0:
        return {
            "matching_bits": 0,
            "compared_bits": 0,
            "hamming_distance": 0,
            "match_percentage": 0.0
        }

    ref = reference[:length]
    rec = recovered[:length]

    matches = int(
        np.sum(ref == rec)
    )

    distance = int(
        np.sum(ref != rec)
    )

    percentage = (
        matches / length
    ) * 100.0

    return {
        "matching_bits": matches,
        "compared_bits": length,
        "hamming_distance": distance,
        "match_percentage": round(
            percentage,
            4
        )
    }


# ============================================================
# REFERENCE-BASED DETECTION
# ============================================================

def detect_interleaver_from_reference(
    reference_bits,
    observed_bits,
    rows=8,
    branches=4,
    delay=3,
    seed=42
):
    """
    Identify a known interleaver by comparing the original
    reference bitstream with the observed interleaved stream.

    This detector only claims identification when there is
    measurable evidence from the reference bitstream.
    """

    reference = _validate_bits(
        reference_bits
    )

    observed = _validate_bits(
        observed_bits
    )

    candidates = []

    # --------------------------------------------------------
    # BLOCK
    # --------------------------------------------------------

    try:

        recovered = block_deinterleave(
            observed,
            rows=rows
        )

        score = _bit_match_score(
            reference,
            recovered
        )

        candidates.append({
            "type": "BLOCK",
            "parameters": {
                "rows": rows
            },
            **score
        })

    except Exception:
        pass

    # --------------------------------------------------------
    # CONVOLUTIONAL
    # --------------------------------------------------------

    try:

        recovered = convolutional_deinterleave(
            observed,
            branches=branches,
            delay=delay
        )

        score = _bit_match_score(
            reference,
            recovered
        )

        candidates.append({
            "type": "CONVOLUTIONAL",
            "parameters": {
                "branches": branches,
                "delay": delay
            },
            **score
        })

    except Exception:
        pass

    # --------------------------------------------------------
    # DIAGONAL
    # --------------------------------------------------------

    try:

        recovered = diagonal_deinterleave(
            observed,
            rows=rows
        )

        score = _bit_match_score(
            reference,
            recovered
        )

        candidates.append({
            "type": "DIAGONAL",
            "parameters": {
                "rows": rows
            },
            **score
        })

    except Exception:
        pass

    # --------------------------------------------------------
    # PSEUDO-RANDOM
    # --------------------------------------------------------

    try:

        recovered = pseudo_random_deinterleave(
            observed,
            seed=seed
        )

        score = _bit_match_score(
            reference,
            recovered
        )

        candidates.append({
            "type": "PSEUDO-RANDOM",
            "parameters": {
                "seed": seed
            },
            **score
        })

    except Exception:
        pass

    # --------------------------------------------------------
    # NO VALID CANDIDATES
    # --------------------------------------------------------

    if not candidates:

        return {
            "available": True,
            "detected": False,
            "type": "NONE",
            "confidence": 0.0,
            "reason": (
                "No compatible interleaver "
                "configuration was found."
            ),
            "candidates": []
        }

    # --------------------------------------------------------
    # SORT BY MATCH PERCENTAGE
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[
            "match_percentage"
        ],
        reverse=True
    )

    best = candidates[0]

    second_score = (
        candidates[1]["match_percentage"]
        if len(candidates) > 1
        else 0.0
    )

    margin = (
        best["match_percentage"]
        - second_score
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

# --------------------------------------------------------
# CONFIDENCE
# --------------------------------------------------------

    if best["match_percentage"] >= 99.99:
        confidence = 100.0

    else:
        confidence = (
            best["match_percentage"]
            * 0.9
            + max(margin, 0.0)
            * 0.1
        )

        confidence = min(
            100.0,
            max(
                0.0,
                confidence
            )
        )

    # --------------------------------------------------------
    # DETECTION DECISION
    # --------------------------------------------------------

    detected = (
        best["match_percentage"] >= 95.0
        and margin >= 1.0
    )

    if detected:

        detection_type = best["type"]

        reason = (
            "Interleaver identified using "
            "reference-bit correlation."
        )

    else:

        detection_type = (
            "NOT_CONFIDENTLY_IDENTIFIED"
        )

        reason = (
            "Reference correlation did not provide "
            "sufficient evidence for unique identification."
        )

    return {
        "available": True,
        "detected": detected,
        "type": detection_type,
        "confidence": round(
            confidence,
            4
        ),
        "score": round(
            best["match_percentage"],
            4
        ),
        "margin": round(
            margin,
            4
        ),
        "reference_bits": len(reference),
        "observed_bits": len(observed),
        "candidates": candidates,
        "reason": reason
    }