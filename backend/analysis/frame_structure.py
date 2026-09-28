"""
SPECTRA v16
Frame / Header / Payload / CRC analysis.

This module performs conservative structural analysis on a recovered
binary bitstream.

It does NOT claim protocol identification.

It looks for:
    - repeated frame spacing
    - frame consistency
    - byte alignment
    - repeated/sync-like prefix
    - stable header-like bytes
    - payload candidate
    - common CRC candidates
    - HEX / ASCII payload previews
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

MIN_BITS = 64
MIN_FRAMES = 3

MIN_FRAME_BITS = 8
MAX_FRAME_BITS = 4096

MAX_ANALYSIS_BITS = 100_000

MAX_SYNC_BYTES = 4
MAX_HEADER_BYTES = 16

HEADER_AGREEMENT_THRESHOLD = 0.75

FRAME_WEAK_THRESHOLD = 0.62
FRAME_MODERATE_THRESHOLD = 0.72
FRAME_STRONG_THRESHOLD = 0.85

CRC_MIN_VALID_FRAMES = 3
CRC_VALIDATION_THRESHOLD = 0.60


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def _clean_bits(
    bits: Sequence[int] | str,
) -> np.ndarray:
    """
    Convert a recovered bitstream to uint8 binary values.

    Non-binary characters are ignored.
    """

    if isinstance(bits, str):

        return np.fromiter(
            (
                1 if b == "1" else 0
                for b in bits
                if b in ("0", "1")
            ),
            dtype=np.uint8,
        )

    arr = np.asarray(
        list(bits),
        dtype=np.uint8,
    )

    return arr[
        (arr == 0) |
        (arr == 1)
    ]


# ============================================================
# CRC IMPLEMENTATIONS
# ============================================================

def _crc8_atm(
    data: Sequence[int],
) -> int:
    """
    CRC-8/ATM / CRC-8/ITU
    Polynomial 0x07
    Initial value 0x00
    """

    crc = 0

    for value in data:

        crc ^= int(value)

        for _ in range(8):

            if crc & 0x80:

                crc = (
                    (crc << 1) ^
                    0x07
                ) & 0xFF

            else:

                crc = (
                    crc << 1
                ) & 0xFF

    return crc


def _crc16_ccitt_false(
    data: Sequence[int],
) -> int:
    """
    CRC-16/CCITT-FALSE
    Polynomial 0x1021
    Initial 0xFFFF
    """

    crc = 0xFFFF

    for value in data:

        crc ^= int(value) << 8

        for _ in range(8):

            if crc & 0x8000:

                crc = (
                    (crc << 1) ^
                    0x1021
                ) & 0xFFFF

            else:

                crc = (
                    crc << 1
                ) & 0xFFFF

    return crc


def _crc16_ibm(
    data: Sequence[int],
) -> int:
    """
    CRC-16/IBM / ARC
    Polynomial 0xA001
    Initial value 0x0000
    Reflected implementation.
    """

    crc = 0x0000

    for value in data:

        crc ^= int(value)

        for _ in range(8):

            if crc & 1:

                crc = (
                    crc >> 1
                ) ^ 0xA001

            else:

                crc >>= 1

            crc &= 0xFFFF

    return crc


def _crc32_ieee(
    data: Sequence[int],
) -> int:
    """
    CRC-32/IEEE 802.3.

    Implemented directly to keep this module dependency-free.
    """

    crc = 0xFFFFFFFF

    for value in data:

        crc ^= int(value)

        for _ in range(8):

            if crc & 1:

                crc = (
                    crc >> 1
                ) ^ 0xEDB88320

            else:

                crc >>= 1

            crc &= 0xFFFFFFFF

    return crc ^ 0xFFFFFFFF


# ============================================================
# BYTE HELPERS
# ============================================================

def _bits_to_bytes(
    bits: np.ndarray,
) -> List[int]:
    """
    Convert a byte-aligned bit array into integer bytes.

    Any incomplete trailing bits are ignored.
    """

    usable = len(bits) - (len(bits) % 8)
    if usable <= 0:

        return []

    reshaped = bits[:usable].reshape(
        -1,
        8,
    )

    weights = np.array(
        [128, 64, 32, 16, 8, 4, 2, 1],
        dtype=np.uint16,
    )

    values = (
        reshaped.astype(np.uint16) *
        weights
    ).sum(axis=1)

    return [
        int(x)
        for x in values
    ]


def _hex_preview(
    data: Sequence[int],
    maximum: int = 128,
) -> str:

    if not data:

        return ""

    return " ".join(
        f"{int(x):02X}"
        for x in data[:maximum]
    )


def _ascii_preview(
    data: Sequence[int],
    maximum: int = 128,
) -> str:

    if not data:

        return ""

    out = []

    for value in data[:maximum]:

        value = int(value)

        if 32 <= value <= 126:

            out.append(
                chr(value)
            )

        else:

            out.append(".")

    return "".join(out)


# ============================================================
# AUTOCORRELATION / FRAME PERIOD SEARCH
# ============================================================

def _period_scores(
    bits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Estimate binary equality agreement for candidate lags.

    Bits are mapped to +/-1.

    mean(x[i] * x[i+lag]) =
        +1 for identical bits
        -1 for opposite bits

    Therefore:
        agreement = (correlation + 1) / 2
    """

    n = len(bits)

    max_lag = min(
        MAX_FRAME_BITS,
        n // 3,
    )

    if max_lag < MIN_FRAME_BITS:

        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    x = np.where(
        bits > 0,
        1.0,
        -1.0,
    )

    fft_length = 1 << (
        (2 * n - 1).bit_length()
    )

    spectrum = np.fft.rfft(
        x,
        fft_length,
    )

    autocorrelation = np.fft.irfft(
        spectrum *
        np.conjugate(spectrum),
        fft_length,
    )[:max_lag + 1]

    lags = np.arange(
        MIN_FRAME_BITS,
        max_lag + 1,
        dtype=np.int32,
    )

    counts = (
        n - lags
    ).astype(np.float64)

    mean_products = (
        autocorrelation[lags] /
        np.maximum(
            counts,
            1.0,
        )
    )

    agreement = (
        mean_products + 1.0
    ) / 2.0

    return lags, agreement


def _local_peaks(
    values: np.ndarray,
) -> np.ndarray:

    if len(values) < 3:

        return np.arange(
            len(values),
            dtype=np.int32,
        )

    middle = values[1:-1]

    indices = np.flatnonzero(
        (
            middle >= values[:-2]
        ) &
        (
            middle >= values[2:]
        )
    ) + 1

    return indices.astype(
        np.int32,
    )


# ============================================================
# FRAME ALIGNMENT
# ============================================================

def _block_consistency(
    bits: np.ndarray,
    frame_length: int,
    offset: int,
) -> Tuple[
    float,
    int,
    List[np.ndarray],
]:
    """
    Partition the bitstream into repeated frame blocks.

    Returns:
        consistency
        number_of_frames
        frame_arrays
    """

    available = (
        len(bits) - offset
    )

    frames = available // frame_length

    if frames < MIN_FRAMES:

        return (
            0.0,
            frames,
            [],
        )

    usable = (
        frames *
        frame_length
    )

    trimmed = bits[
        offset:
        offset + usable
    ]

    frame_array = trimmed.reshape(
        frames,
        frame_length,
    )

    reference = frame_array[0]

    agreement_sum = 0.0
    comparisons = 0

    for row in frame_array[1:]:

        agreement_sum += float(
            np.mean(
                row == reference
            )
        )

        comparisons += 1

    if comparisons == 0:

        return (
            0.0,
            frames,
            [],
        )

    consistency = (
        agreement_sum /
        comparisons
    )

    return (
        float(consistency),
        frames,
        [
            row.copy()
            for row in frame_array
        ],
    )


def _find_best_alignment(
    bits: np.ndarray,
    frame_length: int,
) -> Tuple[
    float,
    int,
    int,
    List[np.ndarray],
]:
    """
    Search a bounded range of frame-start offsets.

    This avoids assuming the first recovered bit is the
    beginning of a frame.
    """

    maximum_offset = min(
        frame_length - 1,
        64,
    )

    best = (
        0.0,
        0,
        0,
        [],
    )

    for offset in range(
        maximum_offset + 1
    ):

        consistency, frames, frame_list = (
            _block_consistency(
                bits,
                frame_length,
                offset,
            )
        )

        if frames < MIN_FRAMES:

            continue

        candidate = (
            consistency,
            frames,
        )

        current_best = (
            best[0],
            best[2],
        )

        if candidate > current_best:

            best = (
                consistency,
                offset,
                frames,
                frame_list,
            )

    return best


# ============================================================
# PREFIX / HEADER ANALYSIS
# ============================================================

def _byte_column_agreement(
    frames: Sequence[Sequence[int]],
    index: int,
) -> float:

    values = []

    for frame in frames:

        if index >= len(frame):

            continue

        values.append(
            int(frame[index])
        )

    if not values:

        return 0.0

    reference = values[0]

    return float(
        sum(
            value == reference
            for value in values
        ) /
        len(values)
    )


def _detect_prefix_and_header(
    frames: Sequence[Sequence[int]],
) -> Tuple[
    int,
    int,
    List[int],
    float,
]:
    """
    Detect a repeated prefix and a stable header-like region.

    Returns:
        sync_bytes
        header_bytes
        header_reference
        header_agreement
    """

    if not frames:

        return (
            0,
            0,
            [],
            0.0,
        )

    bytes_per_frame = len(
        frames[0]
    )

    sync_bytes = 0

    maximum_sync = min(
        MAX_SYNC_BYTES,
        bytes_per_frame,
    )

    for index in range(
        maximum_sync
    ):

        agreement = (
            _byte_column_agreement(
                frames,
                index,
            )
        )

        if agreement >= HEADER_AGREEMENT_THRESHOLD:

            sync_bytes += 1

        else:

            break

    header_bytes = 0
    header_values = []

    maximum_header_end = min(
        sync_bytes +
        MAX_HEADER_BYTES,
        bytes_per_frame,
    )

    agreements = []

    for index in range(
        sync_bytes,
        maximum_header_end,
    ):

        agreement = (
            _byte_column_agreement(
                frames,
                index,
            )
        )

        agreements.append(
            agreement
        )

        if agreement >= HEADER_AGREEMENT_THRESHOLD:

            values = [
                int(frame[index])
                for frame in frames
                if index < len(frame)
            ]

            if values:

                counts: Dict[int, int] = {}

                for value in values:

                    counts[value] = (
                        counts.get(
                            value,
                            0,
                        ) + 1
                    )

                reference = max(
                    counts,
                    key=counts.get,
                )

                header_values.append(
                    reference
                )

            header_bytes += 1

        else:

            # Stop after the first unstable header field.
            break

    if agreements:

        header_agreement = float(
            np.mean(
                agreements[:header_bytes]
            )
        ) if header_bytes else 0.0

    else:

        header_agreement = 0.0

    return (
        sync_bytes,
        header_bytes,
        header_values,
        header_agreement,
    )


# ============================================================
# CRC CANDIDATE TESTING
# ============================================================

def _test_crc_candidate(
    frames: Sequence[Sequence[int]],
    name: str,
    crc_size: int,
    crc_function,
    byteorder: str = "big",
) -> Optional[Dict[str, Any]]:
    """
    Test a CRC candidate assuming the CRC occupies the final
    crc_size bytes of each frame.

    This is intentionally conservative.
    """

    if not frames:

        return None

    minimum_length = (
        crc_size + 1
    )

    if len(frames[0]) < minimum_length:

        return None

    valid = 0
    total = 0

    for frame in frames:

        if len(frame) < minimum_length:

            continue

        data = frame[:-crc_size]
        received = frame[-crc_size:]

        calculated = int(
            crc_function(data)
        )

        if crc_size == 1:

            expected = int(
                received[0]
            )

        elif crc_size == 2:

            if byteorder == "big":

                expected = (
                    int(received[0]) << 8
                ) | int(received[1])

            else:

                expected = (
                    int(received[1]) << 8
                ) | int(received[0])

        elif crc_size == 4:

            if byteorder == "big":

                expected = (
                    (int(received[0]) << 24)
                    |
                    (int(received[1]) << 16)
                    |
                    (int(received[2]) << 8)
                    |
                    int(received[3])
                )

            else:

                expected = (
                    (int(received[3]) << 24)
                    |
                    (int(received[2]) << 16)
                    |
                    (int(received[1]) << 8)
                    |
                    int(received[0])
                )

        else:

            continue

        total += 1

        if calculated == expected:

            valid += 1

    if total < CRC_MIN_VALID_FRAMES:

        return None

    score = (
        valid /
        total
    )

    return {
        "name": name,
        "valid_frames": valid,
        "tested_frames": total,
        "score": float(score),
        "crc_bytes": crc_size,
    }


def _detect_crc(
    frames: Sequence[Sequence[int]],
) -> Optional[Dict[str, Any]]:

    candidates = []

    tests = [
        (
            "CRC-8/ATM",
            1,
            _crc8_atm,
            "big",
        ),
        (
            "CRC-16/CCITT-FALSE",
            2,
            _crc16_ccitt_false,
            "big",
        ),
        (
            "CRC-16/IBM",
            2,
            _crc16_ibm,
            "little",
        ),
        (
            "CRC-32/IEEE",
            4,
            _crc32_ieee,
            "big",
        ),
    ]

    for (
        name,
        crc_size,
        function,
        byteorder,
    ) in tests:

        result = _test_crc_candidate(
            frames,
            name,
            crc_size,
            function,
            byteorder,
        )

        if result is not None:

            candidates.append(
                result
            )

    if not candidates:

        return None

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["valid_frames"],
        ),
        reverse=True,
    )

    best = candidates[0]

    if (
        best["valid_frames"]
        <
        CRC_MIN_VALID_FRAMES
        or
        best["score"]
        <
        CRC_VALIDATION_THRESHOLD
    ):

        return None

    return best


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_frame_structure(
    bits: Sequence[int] | str,
) -> Dict[str, Any]:
    """
    Analyze a recovered bitstream for repeated frame structure.

    The output is deliberately conservative.

    It can return:
        NO_BITSTREAM
        INSUFFICIENT_DATA
        NO_FRAME_STRUCTURE
        FRAME_CANDIDATE
        STRONG_FRAME
    """

    clean = _clean_bits(
        bits
    )

    if len(clean) == 0:

        return {
            "available": False,
            "detected": False,
            "status": "NO_BITSTREAM",
            "reason": (
                "No recovered binary bitstream "
                "is available."
            ),
        }

    original_length = len(clean)

    if len(clean) < MIN_BITS:

        return {
            "available": True,
            "detected": False,
            "status": "INSUFFICIENT_DATA",
            "reason": (
                f"Need at least {MIN_BITS} recovered bits "
                "for frame analysis."
            ),
            "input_bits": int(
                original_length
            ),
        }

    # Bound processing time for large captures.
    if len(clean) > MAX_ANALYSIS_BITS:

        clean = clean[
            :MAX_ANALYSIS_BITS
        ]

    input_bits = len(clean)

    lags, agreements = _period_scores(
        clean
    )

    if len(lags) == 0:

        return {
            "available": True,
            "detected": False,
            "status": "NO_FRAME_STRUCTURE",
            "reason": (
                "Recovered bitstream is too short "
                "for repeated-frame analysis."
            ),
            "input_bits": int(
                input_bits
            ),
        }

    peak_indices = _local_peaks(
        agreements
    )

    candidate_indices = [
        index
        for index in peak_indices
        if agreements[index]
        >= FRAME_WEAK_THRESHOLD
    ]

    # Also retain the absolute best lag.
    if len(agreements):

        best_index = int(
            np.argmax(
                agreements
            )
        )

        if best_index not in candidate_indices:

            candidate_indices.append(
                best_index
            )

    candidate_results = []

    for index in candidate_indices:

        if index < 0 or index >= len(lags):

            continue

        frame_length = int(
            lags[index]
        )

        if frame_length < MIN_FRAME_BITS:

            continue

        consistency, offset, frames, frame_list = (
            _find_best_alignment(
                clean,
                frame_length,
            )
        )

        if frames < MIN_FRAMES:

            continue

        if consistency < FRAME_WEAK_THRESHOLD:

            continue

        candidate_results.append(
            {
                "frame_length_bits": frame_length,
                "consistency": consistency,
                "offset_bits": offset,
                "frames": frames,
                "frames_data": frame_list,
            }
        )

    if not candidate_results:

        return {
            "available": True,
            "detected": False,
            "status": "NO_FRAME_STRUCTURE",
            "reason": (
                "No strong repeated frame spacing "
                "was detected in the recovered bitstream."
            ),
            "input_bits": int(
                input_bits
            ),
        }

    # Prefer:
    #   1. consistency
    #   2. byte alignment
    #   3. number of frames
    def candidate_score(
        item: Dict[str, Any],
    ):

        alignment_bonus = (
            0.01
            if item[
                "frame_length_bits"
            ] % 8 == 0
            else 0.0
        )

        return (
            item["consistency"]
            +
            alignment_bonus,
            item["frames"],
        )

    best = max(
        candidate_results,
        key=candidate_score,
    )

    frame_length_bits = int(
        best["frame_length_bits"]
    )

    frame_length_bytes = (
        frame_length_bits // 8
        if frame_length_bits % 8 == 0
        else None
    )

    consistency = float(
        best["consistency"]
    )

    frame_count = int(
        best["frames"]
    )

    offset_bits = int(
        best["offset_bits"]
    )

    status = (
        "STRONG_FRAME"
        if consistency >= FRAME_STRONG_THRESHOLD
        else
        "FRAME_CANDIDATE"
    )

    result: Dict[str, Any] = {
        "available": True,
        "detected": True,
        "status": status,
        "reason": (
            "Repeated frame structure detected "
            "from recovered bitstream consistency."
        ),
        "input_bits": int(
            input_bits
        ),
        "analysis_bits": int(
            input_bits
        ),
        "frame_length_bits": frame_length_bits,
        "frame_length_bytes": frame_length_bytes,
        "frames_detected": frame_count,
        "frame_consistency": consistency,
        "frame_offset_bits": offset_bits,
        "byte_aligned": (
            frame_length_bits % 8 == 0
        ),
    }

    # --------------------------------------------------------
    # BYTE-LEVEL PAYLOAD ANALYSIS
    # --------------------------------------------------------

    if (
        frame_length_bytes is None
        or frame_length_bytes < 2
    ):

        result.update(
            {
                "payload_analysis_available": False,
                "payload_reason": (
                    "Estimated frame is not "
                    "byte-aligned."
                ),
                "sync_word_bits": "",
                "sync_word_hex": "",
                "header_candidate": None,
                "payload_candidate": None,
                "crc_candidate": None,
            }
        )

        return result

    frame_bytes = []

    for frame_bits in best[
        "frames_data"
    ]:

        converted = _bits_to_bytes(
            frame_bits
        )

        if len(converted) != frame_length_bytes:

            continue

        frame_bytes.append(
            converted
        )

    if len(frame_bytes) < MIN_FRAMES:

        result.update(
            {
                "payload_analysis_available": False,
                "payload_reason": (
                    "Not enough complete byte-aligned "
                    "frames for header/payload analysis."
                ),
                "sync_word_bits": "",
                "sync_word_hex": "",
                "header_candidate": None,
                "payload_candidate": None,
                "crc_candidate": None,
            }
        )

        return result

    (
        sync_bytes,
        header_bytes,
        header_reference,
        header_agreement,
    ) = _detect_prefix_and_header(
        frame_bytes
    )

    sync_reference = []

    if sync_bytes > 0:

        sync_reference = frame_bytes[0][
            :sync_bytes
        ]

    sync_bits = ""

    if sync_reference:

        sync_bits = "".join(
            f"{value:08b}"
            for value in sync_reference
        )

    crc_candidate = _detect_crc(
        frame_bytes
    )

    crc_bytes = (
        int(
            crc_candidate[
                "crc_bytes"
            ]
        )
        if crc_candidate
        else 0
    )

    payload_start = (
        sync_bytes +
        header_bytes
    )

    payload_bytes = max(
        0,
        frame_length_bytes -
        payload_start -
        crc_bytes,
    )

    first_frame = frame_bytes[0]

    header_data = (
        first_frame[
            sync_bytes:
            sync_bytes + header_bytes
        ]
        if header_bytes
        else []
    )

    payload_data = (
        first_frame[
            payload_start:
            payload_start + payload_bytes
        ]
        if payload_bytes
        else []
    )

    payload_confidence_parts = [
        consistency
    ]
    payload_weights = [
        0.65
    ]

    if header_bytes > 0:

        payload_confidence_parts.append(
            header_agreement
        )
        payload_weights.append(
            0.25
        )

    if crc_candidate:

        payload_confidence_parts.append(
            float(
                crc_candidate[
                    "score"
                ]
            )
        )
        payload_weights.append(
            0.10
        )

    denominator = sum(
        payload_weights
    )

    confidence = (
        sum(
            value * weight
            for value, weight in zip(
                payload_confidence_parts,
                payload_weights,
            )
        )
        /
        denominator
    )

    result.update(
        {
            "payload_analysis_available": True,
            "payload_reason": (
                "Header/payload are heuristic candidates "
                "derived from repeated frame structure."
            ),
            "sync_word_bits": sync_bits,
            "sync_word_hex": _hex_preview(
                sync_reference
            ),
            "sync_bytes": sync_bytes,
            "header_candidate": {
                "offset_bytes": sync_bytes,
                "length_bytes": header_bytes,
                "agreement": header_agreement,
                "hex": _hex_preview(
                    header_data
                ),
            },
            "payload_candidate": {
                "offset_bytes": payload_start,
                "length_bytes": payload_bytes,
                "confidence": float(
                    confidence
                ),
                "hex_preview": _hex_preview(
                    payload_data
                ),
                "ascii_preview": _ascii_preview(
                    payload_data
                ),
            },
            "crc_candidate": crc_candidate,
            "confidence": float(
                confidence
            ),
        }
    )

    return result