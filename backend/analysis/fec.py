import numpy as np

from backend.analysis.ldpc_candidate_search import auto_ldpc_search


# ============================================================
# SPECTRA — FEC / ERROR-CONTROL ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Analyze a recovered bitstream for evidence of FEC and,
# when enough information is available, attempt decoding.
#
# Required PS FEC families:
#
#   1. Short-constraint convolutional codes + Viterbi
#   2. Reed-Solomon block codes
#   3. Concatenated codes
#   4. LDPC
#
# IMPORTANT
# ---------
# A completely unknown RF bitstream does not contain enough
# information to uniquely determine every possible FEC code.
#
# Therefore SPECTRA:
#
#   - measures observable evidence
#   - tests bounded supported candidates
#   - validates candidates before claiming them
#   - decodes only when a candidate can be validated
#   - otherwise reports uncertainty
#
# Existing output states are preserved:
#
#   NONE
#   POSSIBLE
#   LIKELY
# ============================================================


# ============================================================
# OPTIONAL DECODER LIBRARIES
# ============================================================

try:
    import reedsolo

    REEDSOLO_AVAILABLE = True

except Exception:
    reedsolo = None
    REEDSOLO_AVAILABLE = False


SUPPORTED_FEC_FAMILIES = [
    "Convolutional + Viterbi",
    "Reed-Solomon",
    "Concatenated",
    "LDPC",
]


# ============================================================
# INPUT CLEANING
# ============================================================

def _clean_bits(bits):
    """
    Convert an input bitstream into a uint8 array containing
    only 0 and 1.
    """

    if bits is None:
        return np.zeros(
            0,
            dtype=np.uint8
        )

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
# BASIC BIT STATISTICS
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
# PARITY / CHECK-BIT EVIDENCE
# ============================================================

def _parity_evidence(bits):

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

        data = x[
            :,
            :-1
        ]

        parity = x[
            :,
            -1
        ]

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

        evidence = (
            abs(
                consistency
                -
                0.50
            )
            *
            2.0
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
# REPETITION EVIDENCE
# ============================================================

def _repetition_evidence(bits):

    n = len(bits)

    if n < 12:

        return {
            "best_repeat": None,
            "repetition_score": 0.0,
            "copy_agreement": 0.0,
        }

    best_repeat = None
    best_score = 0.0
    best_copy_agreement = 0.0

    # --------------------------------------------------------
    # Bit-level repetition
    # --------------------------------------------------------

    for repeat in (
        2,
        3,
        4,
    ):

        usable = (
            n
            //
            repeat
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

    # --------------------------------------------------------
    # Whole-block repetition
    # --------------------------------------------------------

    for repeat in (
        2,
        3,
        4,
    ):

        block_size = (
            n
            //
            repeat
        )

        if block_size < 16:
            continue

        usable = (
            block_size
            *
            repeat
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
                    x[0]
                    ==
                    x[k]
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

    if best_score < 0.70:

        return {
            "best_repeat": None,
            "repetition_score": 0.0,
            "copy_agreement": round(
                best_score,
                4
            ),
        }

    repetition_score = float(
        np.clip(
            (
                best_score
                -
                0.50
            )
            /
            0.50,
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
# HAMMING-LIKE SUPPORTING EVIDENCE
# ============================================================

def _hamming74_evidence(bits):

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

        usable = x[
            :n_blocks * block_size
        ].reshape(
            n_blocks,
            block_size
        )

        correlations = []

        for k in range(
            n_blocks - 1
        ):

            a = (
                usable[k]
                -
                np.mean(
                    usable[k]
                )
            )

            b = (
                usable[k + 1]
                -
                np.mean(
                    usable[k + 1]
                )
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
# CODE-RATE / LENGTH CLUES
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
# GF(256) SUPPORT FOR REED-SOLOMON ANALYSIS
# ============================================================

def _gf_tables():

    exp = np.zeros(
        512,
        dtype=np.int16
    )

    log = np.full(
        256,
        -1,
        dtype=np.int16
    )

    x = 1

    for i in range(255):

        exp[i] = x
        log[x] = i

        x <<= 1

        if x & 0x100:
            x ^= 0x11D

    for i in range(
        255,
        512
    ):

        exp[i] = exp[
            i - 255
        ]

    return exp, log


_GF_EXP, _GF_LOG = _gf_tables()


def _gf_mul(a, b):

    a = int(a)
    b = int(b)

    if a == 0 or b == 0:
        return 0

    return int(
        _GF_EXP[
            int(_GF_LOG[a])
            +
            int(_GF_LOG[b])
        ]
    )


def _gf_pow(a, n):

    if n == 0:
        return 1

    if a == 0:
        return 0

    exponent = (
        int(_GF_LOG[a])
        *
        int(n)
    ) % 255

    return int(
        _GF_EXP[
            exponent
        ]
    )


def _rs_syndromes(codeword, nsym):

    syndromes = []

    for i in range(
        nsym
    ):

        root = _GF_EXP[i]

        value = 0

        for byte in codeword:

            value = (
                _gf_mul(
                    value,
                    root
                )
                ^
                int(byte)
            )

        syndromes.append(
            value
        )

    return np.asarray(
        syndromes,
        dtype=np.uint8
    )


# ============================================================
# BITS → BYTES
# ============================================================

def _bits_to_bytes(bits, offset=0):

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    if (
        offset < 0
        or
        offset >= 8
    ):

        return np.zeros(
            0,
            dtype=np.uint8
        )

    usable = (
        (
            len(bits)
            -
            offset
        )
        //
        8
    ) * 8

    if usable <= 0:

        return np.zeros(
            0,
            dtype=np.uint8
        )

    x = bits[
        offset:
        offset + usable
    ].reshape(
        -1,
        8
    )

    weights = np.array(
        [
            128,
            64,
            32,
            16,
            8,
            4,
            2,
            1
        ],
        dtype=np.uint16
    )

    values = (
        x
        *
        weights
    ).sum(
        axis=1
    )

    return values.astype(
        np.uint8
    )


def _bytes_to_bits(data):

    data = np.asarray(
        data,
        dtype=np.uint8
    )

    if len(data) == 0:
        return np.zeros(
            0,
            dtype=np.uint8
        )

    bits = np.unpackbits(
        data
    )

    return bits.astype(
        np.uint8
    )


# ============================================================
# REED-SOLOMON EVIDENCE
# ============================================================

def _reed_solomon_evidence(bits):

    result = {
        "available": False,
        "best_nsym": None,
        "best_offset": None,
        "codeword_size": 255,
        "syndrome_zero_rate": 0.0,
        "codewords_tested": 0,
        "valid_codewords": 0,
        "evidence": 0.0,
        "decoder_available": REEDSOLO_AVAILABLE,
    }

    if len(bits) < 2040:
        return result

    best = None

    for offset in range(8):

        raw = _bits_to_bytes(
            bits,
            offset
        )

        if len(raw) < 255:
            continue

        usable = (
            len(raw)
            //
            255
        ) * 255

        codewords = raw[
            :usable
        ].reshape(
            -1,
            255
        )

        for nsym in (
            8,
            16,
            24,
            32,
        ):

            valid_count = 0

            for codeword in codewords:

                syndrome = _rs_syndromes(
                    codeword,
                    nsym
                )

                if np.all(
                    syndrome == 0
                ):

                    valid_count += 1

            total = len(
                codewords
            )

            if total == 0:
                continue

            valid_rate = (
                valid_count
                /
                total
            )

            # A valid RS codeword must satisfy all checked
            # syndromes. Even one fully valid codeword is
            # significant, while multiple valid codewords
            # provide much stronger evidence.
            if best is None:

                best = (
                    valid_rate,
                    offset,
                    nsym,
                    total,
                    valid_count
                )

            else:

                if valid_rate > best[0]:

                    best = (
                        valid_rate,
                        offset,
                        nsym,
                        total,
                        valid_count
                    )

    if best is None:
        return result

    (
        valid_rate,
        offset,
        nsym,
        total,
        valid_count
    ) = best

    if valid_rate >= 0.75:

        evidence = 1.0

    elif valid_rate >= 0.50:

        evidence = 0.80

    elif valid_rate >= 0.25:

        evidence = 0.50

    elif valid_count > 0:

        evidence = 0.25

    else:

        evidence = 0.0

    result.update(
        {
            "available": True,
            "best_nsym": int(nsym),
            "best_offset": int(offset),
            "syndrome_zero_rate": round(
                float(valid_rate),
                4
            ),
            "codewords_tested": int(total),
            "valid_codewords": int(valid_count),
            "evidence": round(
                float(evidence),
                4
            ),
        }
    )

    return result


# ============================================================
# REED-SOLOMON DECODING
# ============================================================

def _try_reed_solomon_decode(
    bits,
    rs_info
):

    result = {
        "status": "NOT_ATTEMPTED",
        "corrected_bits": None,
        "corrections": 0,
        "details": {},
    }

    if not rs_info.get(
        "available",
        False
    ):

        return result

    if not REEDSOLO_AVAILABLE:

        result[
            "status"
        ] = "DECODER_LIBRARY_UNAVAILABLE"

        result[
            "details"
        ] = {
            "message":
                "reedsolo is not installed"
        }

        return result

    offset = int(
        rs_info[
            "best_offset"
        ]
    )

    nsym = int(
        rs_info[
            "best_nsym"
        ]
    )

    raw = _bits_to_bytes(
        bits,
        offset
    )

    if len(raw) < 255:

        result[
            "status"
        ] = "INSUFFICIENT_DATA"

        return result

    usable = (
        len(raw)
        //
        255
    ) * 255

    codewords = raw[
        :usable
    ].reshape(
        -1,
        255
    )

    corrected = []

    corrected_count = 0
    successful = 0

    try:

        codec = reedsolo.RSCodec(
            nsym,
            nsize=255
        )

    except Exception as exc:

        result[
            "status"
        ] = "DECODER_INIT_FAILED"

        result[
            "details"
        ] = {
            "message": str(exc)
        }

        return result

    for codeword in codewords:

        cw = bytes(
            int(v)
            for v in codeword
        )

        try:

            decoded = codec.decode(
                cw
            )

            if isinstance(
                decoded,
                tuple
            ):

                message = decoded[0]

            else:

                message = decoded

            message = bytes(
                message
            )

            # For display/correction purposes we return
            # the decoded message bytes, not the encoded
            # parity bytes.
            corrected.extend(
                message
            )

            successful += 1

        except Exception:

            # Keep original bytes if this codeword cannot
            # be corrected.
            corrected.extend(
                cw
            )

    if successful == 0:

        result[
            "status"
        ] = "FAILED"

        return result

    corrected_array = np.asarray(
        list(corrected),
        dtype=np.uint8
    )

    corrected_bits = _bytes_to_bits(
        corrected_array
    )

    result[
        "status"
    ] = "SUCCESS"

    result[
        "corrected_bits"
    ] = "".join(
        corrected_bits.astype(
            str
        )
    )

    result[
        "corrections"
    ] = corrected_count

    result[
        "details"
    ] = {
        "offset": offset,
        "nsym": nsym,
        "codeword_size": 255,
        "codewords_decoded": int(
            successful
        ),
        "decoder": "Reed-Solomon",
    }

    return result


# ============================================================
# CONVOLUTIONAL CODE DEFINITIONS
# ============================================================

CONVOLUTIONAL_CANDIDATES = [
    {
        "constraint_length": 3,
        "generators": (
            0o7,
            0o5
        ),
        "name":
            "K=3, generators=(7,5)"
    },
    {
        "constraint_length": 5,
        "generators": (
            0o23,
            0o35
        ),
        "name":
            "K=5, generators=(23,35)"
    },
    {
        "constraint_length": 7,
        "generators": (
            0o171,
            0o133
        ),
        "name":
            "K=7, generators=(171,133)"
    },
]


def _parity(value):

    return int(
        value.bit_count()
        &
        1
    )


def _conv_next(
    state,
    bit,
    constraint_length,
    generators
):

    register = (
        (
            state
            <<
            1
        )
        |
        int(bit)
    )

    mask = (
        1
        <<
        constraint_length
    ) - 1

    register &= mask

    next_state = (
        register
        &
        (
            (
                1
                <<
                (
                    constraint_length
                    -
                    1
                )
            )
            -
            1
        )
    )

    output = tuple(
        _parity(
            register
            &
            int(generator)
        )
        for generator in generators
    )

    return (
        next_state,
        output
    )


def _conv_encode(
    bits,
    constraint_length,
    generators
):

    state = 0
    output = []

    for bit in bits:

        state, encoded = _conv_next(
            state,
            int(bit),
            constraint_length,
            generators
        )

        output.extend(
            encoded
        )

    return np.asarray(
        output,
        dtype=np.uint8
    )


# ============================================================
# VITERBI DECODER
# ============================================================

def _viterbi_decode(
    received,
    constraint_length,
    generators
):

    received = np.asarray(
        received,
        dtype=np.uint8
    )

    n_outputs = len(
        generators
    )

    if (
        n_outputs == 0
        or
        len(received)
        <
        n_outputs
    ):

        return (
            None,
            np.inf
        )

    usable = (
        len(received)
        //
        n_outputs
    ) * n_outputs

    received = received[
        :usable
    ]

    steps = (
        len(received)
        //
        n_outputs
    )

    n_states = (
        1
        <<
        (
            constraint_length
            -
            1
        )
    )

    metrics = np.full(
        n_states,
        np.inf,
        dtype=np.float64
    )

    # A normal convolutional encoder starts in state 0.
    metrics[0] = 0.0

    previous_states = np.zeros(
        (
            steps,
            n_states
        ),
        dtype=np.int16
    )

    previous_bits = np.zeros(
        (
            steps,
            n_states
        ),
        dtype=np.uint8
    )

    for step in range(steps):

        r = received[
            step * n_outputs:
            (step + 1) * n_outputs
        ]

        new_metrics = np.full(
            n_states,
            np.inf,
            dtype=np.float64
        )

        for state in range(
            n_states
        ):

            current_metric = metrics[
                state
            ]

            if not np.isfinite(
                current_metric
            ):

                continue

            for bit in (
                0,
                1
            ):

                next_state, expected = _conv_next(
                    state,
                    bit,
                    constraint_length,
                    generators
                )

                branch_metric = sum(
                    int(
                        a != b
                    )
                    for a, b
                    in zip(
                        r,
                        expected
                    )
                )

                candidate_metric = (
                    current_metric
                    +
                    branch_metric
                )

                if (
                    candidate_metric
                    <
                    new_metrics[
                        next_state
                    ]
                ):

                    new_metrics[
                        next_state
                    ] = candidate_metric

                    previous_states[
                        step,
                        next_state
                    ] = state

                    previous_bits[
                        step,
                        next_state
                    ] = bit

        metrics = new_metrics

    final_state = int(
        np.argmin(
            metrics
        )
    )

    path_metric = float(
        metrics[
            final_state
        ]
    )

    decoded = np.zeros(
        steps,
        dtype=np.uint8
    )

    state = final_state

    for step in range(
        steps - 1,
        -1,
        -1
    ):

        decoded[
            step
        ] = previous_bits[
            step,
            state
        ]

        state = int(
            previous_states[
                step,
                state
            ]
        )

    return (
        decoded,
        path_metric
    )


# ============================================================
# CONVOLUTIONAL CANDIDATE ANALYSIS
# ============================================================

def _convolutional_evidence(bits):

    result = {
        "available": False,
        "best": None,
        "candidates": [],
        "evidence": 0.0,
    }

    if len(bits) < 64:

        return result

    best = None

    for candidate in (
        CONVOLUTIONAL_CANDIDATES
    ):

        K = candidate[
            "constraint_length"
        ]

        generators = candidate[
            "generators"
        ]

        for swap in (
            False,
            True
        ):

            received = bits

            if len(received) % 2:
                received = received[
                    :-1
                ]

            if swap:

                received = received.reshape(
                    -1,
                    2
                )[
                    :,
                    ::-1
                ].reshape(
                    -1
                )

            decoded, path_metric = _viterbi_decode(
                received,
                K,
                generators
            )

            if decoded is None:
                continue

            reencoded = _conv_encode(
                decoded,
                K,
                generators
            )

            usable = min(
                len(received),
                len(reencoded)
            )

            if usable == 0:
                continue

            mismatch = float(
                np.mean(
                    received[
                        :usable
                    ]
                    !=
                    reencoded[
                        :usable
                    ]
                )
            )

            consistency = (
                1.0
                -
                mismatch
            )

            # For a rate-1/2 code a random stream should
            # not normally achieve very high consistency.
            evidence = float(
                np.clip(
                    (
                        consistency
                        -
                        0.55
                    )
                    /
                    0.45,
                    0.0,
                    1.0
                )
            )

            entry = {
                "constraint_length": int(K),
                "generators": [
                    int(g)
                    for g in generators
                ],
                "generator_order_swapped": bool(
                    swap
                ),
                "consistency": round(
                    consistency,
                    4
                ),
                "mismatch_rate": round(
                    mismatch,
                    4
                ),
                "path_metric": round(
                    path_metric,
                    4
                ),
                "evidence": round(
                    evidence,
                    4
                ),
                "decoded_bits": "".join(
                    decoded.astype(
                        str
                    )
                ),
            }

            result[
                "candidates"
            ].append(
                entry
            )

            if (
                best is None
                or
                evidence > best[
                    "evidence"
                ]
            ):

                best = entry

    if best is None:

        return result

    result[
        "available"
    ] = True

    result[
        "best"
    ] = best

    result[
        "evidence"
    ] = float(
        best[
            "evidence"
        ]
    )

    return result


# ============================================================
# VITERBI DECODING
# ============================================================

def _try_viterbi_decode(
    bits,
    conv_info
):

    result = {
        "status": "NOT_ATTEMPTED",
        "corrected_bits": None,
        "corrections": 0,
        "details": {},
    }

    if not conv_info.get(
        "available",
        False
    ):

        return result

    best = conv_info.get(
        "best"
    )

    if not best:

        return result

    decoded = best.get(
        "decoded_bits"
    )

    if not decoded:

        result[
            "status"
        ] = "FAILED"

        return result

    consistency = float(
        best.get(
            "consistency",
            0.0
        )
    )

    candidates = conv_info.get(
        "candidates",
        []
    )

    other_evidence = sorted(
        [
            float(
                c.get(
                    "evidence",
                    0.0
                )
            )
            for c in candidates
            if c is not best
        ],
        reverse=True
    )

    second_best_evidence = (
        other_evidence[0]
        if other_evidence
        else 0.0
    )

    evidence_margin = (
        float(
            best.get(
                "evidence",
                0.0
            )
        )
        -
        second_best_evidence
    )

    # Do not claim a validated FEC decode merely because
    # Viterbi can always find a path. Require strong
    # re-encode consistency.
    if (
        consistency < 0.95
        or
        float(
            best.get(
                "evidence",
                0.0
            )
        ) < 0.88
        or
        evidence_margin < 0.05
    ):

        result[
            "status"
        ] = "CANDIDATE_ONLY"

        result[
            "details"
        ] = {
            "reason":
                "Candidate did not reach validation threshold.",
            "consistency":
                consistency,
        }

        return result

    result[
        "status"
    ] = "VALIDATED"

    result[
        "corrected_bits"
    ] = decoded

    result[
        "corrections"
    ] = int(
        round(
            float(
                best.get(
                    "mismatch_rate",
                    0.0
                )
            )
            *
            len(bits)
        )
    )

    result[
        "details"
    ] = {
        "family":
            "Convolutional + Viterbi",
        "constraint_length":
            best[
                "constraint_length"
            ],
        "generators":
            best[
                "generators"
            ],
        "rate":
            "1/2",
        "reencoded_consistency":
            consistency,
        "decoder":
            "Viterbi",
    }

    return result


# ============================================================
# LDPC SUPPORT
# ============================================================

def _ldpc_evidence(
    bits,
    parity_check_matrix=None
):

    result = {
        "available": False,
        "matrix_supplied": False,
        "matrix_shape": None,
        "syndrome_zero_rate": 0.0,
        "evidence": 0.0,
        "message": (
            "LDPC analysis requires a supported "
            "parity-check matrix/profile."
        ),
    }

    if parity_check_matrix is None:

        return result

    H = np.asarray(
        parity_check_matrix,
        dtype=np.uint8
    )

    if H.ndim != 2:

        return result

    if H.shape[1] != len(bits):

        return {
            **result,
            "matrix_supplied": True,
            "matrix_shape": tuple(
                H.shape
            ),
            "message": (
                "Parity-check matrix length "
                "does not match bitstream length."
            ),
        }

    syndrome = (
        (
            H
            @
            bits
        )
        %
        2
    )

    zero_rate = float(
        np.mean(
            syndrome == 0
        )
    )

    if zero_rate >= 0.95:

        evidence = 1.0

    elif zero_rate >= 0.80:

        evidence = 0.75

    elif zero_rate >= 0.60:

        evidence = 0.40

    else:

        evidence = 0.0

    return {
        "available": True,
        "matrix_supplied": True,
        "matrix_shape": tuple(
            H.shape
        ),
        "syndrome_zero_rate": round(
            zero_rate,
            4
        ),
        "evidence": round(
            evidence,
            4
        ),
        "message": (
            "LDPC parity-check evidence calculated."
        ),
    }


def _ldpc_bitflip_decode(
    bits,
    parity_check_matrix,
    max_iterations=20
):

    bits = np.asarray(
        bits,
        dtype=np.uint8
    ).copy()

    H = np.asarray(
        parity_check_matrix,
        dtype=np.uint8
    )

    if (
        H.ndim != 2
        or
        H.shape[1] != len(bits)
    ):

        return {
            "status": "INVALID_MATRIX",
            "corrected_bits": None,
            "corrections": 0,
        }

    corrected_count = 0

    for _ in range(
        max_iterations
    ):

        syndrome = (
            (
                H
                @
                bits
            )
            %
            2
        )

        if np.all(
            syndrome == 0
        ):

            return {
                "status": "VALIDATED",
                "corrected_bits": "".join(
                    bits.astype(
                        str
                    )
                ),
                "corrections":
                    corrected_count,
            }

        unsatisfied_checks = (
            syndrome == 1
        )

        variable_votes = (
            H[
                unsatisfied_checks
            ].sum(
                axis=0
            )
        )

        if len(
            variable_votes
        ) == 0:

            break

        maximum = int(
            np.max(
                variable_votes
            )
        )

        if maximum <= 0:
            break

        threshold = max(
            1,
            int(
                np.ceil(
                    maximum * 0.50
                )
            )
        )

        flip = (
            variable_votes
            >=
            threshold
        )

        if not np.any(
            flip
        ):
            break

        bits[
            flip
        ] ^= 1

        corrected_count += int(
            np.sum(
                flip
            )
        )

    return {
        "status": "FAILED",
        "corrected_bits": None,
        "corrections":
            corrected_count,
    }


# ============================================================
# CONCATENATED DECODING
# ============================================================

def _try_concatenated_decode(
    bits,
    conv_info,
    rs_info
):

    result = {
        "status": "NOT_ATTEMPTED",
        "corrected_bits": None,
        "details": {},
    }

    if not (
        conv_info.get(
            "available",
            False
        )
        and
        rs_info.get(
            "available",
            False
        )
    ):

        return result

    conv = _try_viterbi_decode(
        bits,
        conv_info
    )

    if conv.get(
        "status"
    ) not in (
        "VALIDATED",
    ):

        result[
            "status"
        ] = "CANDIDATE_ONLY"

        return result

    decoded_bits = _clean_bits(
        conv.get(
            "corrected_bits",
            ""
        )
    )

    rs = _try_reed_solomon_decode(
        decoded_bits,
        rs_info
    )

    if rs.get(
        "status"
    ) == "SUCCESS":

        result[
            "status"
        ] = "VALIDATED"

        result[
            "corrected_bits"
        ] = rs[
            "corrected_bits"
        ]

        result[
            "details"
        ] = {
            "structure":
                "Convolutional → Viterbi → Reed-Solomon",
            "inner_decoder":
                "Viterbi",
            "outer_decoder":
                "Reed-Solomon",
        }

    else:

        result[
            "status"
        ] = "CANDIDATE_ONLY"

        result[
            "details"
        ] = {
            "structure":
                "Convolutional + Reed-Solomon",
            "message":
                "Candidate structure found, but "
                "combined decode was not validated."
        }

    return result


# ============================================================
# LEGACY SUPPORTING CLASSIFICATION
# ============================================================

def _legacy_evidence_score(
    parity,
    repetition,
    hamming,
    block
):

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

    block_score = float(
        block.get(
            "best_correlation",
            0.0
        )
    )

    score = 0.0

    # These are supporting observations only.
    if parity_score >= 0.90:
        score += 0.25

    elif parity_score >= 0.75:
        score += 0.15

    elif parity_score >= 0.60:
        score += 0.08

    if repetition_score >= 0.90:
        score += 0.25

    elif repetition_score >= 0.70:
        score += 0.15

    elif repetition_score >= 0.50:
        score += 0.08

    if hamming_score >= 0.80:
        score += 0.25

    elif hamming_score >= 0.50:
        score += 0.15

    elif hamming_score >= 0.30:
        score += 0.08

    if block_score >= 0.90:
        score += 0.10

    elif block_score >= 0.70:
        score += 0.06

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# EVIDENCE REASONS
# ============================================================

def _make_reasons(
    parity,
    repetition,
    hamming,
    block,
    conv,
    rs,
    ldpc,
    concatenated
):

    reasons = []

    # --------------------------------------------------------
    # Convolutional
    # --------------------------------------------------------

    if conv.get(
        "best"
    ):

        best = conv[
            "best"
        ]

        consistency = float(
            best.get(
                "consistency",
                0.0
            )
        )

        if consistency >= 0.80:

            reasons.append(
                "Strong convolutional-code path consistency"
            )

        elif consistency >= 0.65:

            reasons.append(
                "Moderate convolutional-code path consistency"
            )

    # --------------------------------------------------------
    # Reed-Solomon
    # --------------------------------------------------------

    rs_rate = float(
        rs.get(
            "syndrome_zero_rate",
            0.0
        )
    )

    if rs_rate >= 0.75:

        reasons.append(
            "Strong Reed-Solomon syndrome consistency"
        )

    elif rs_rate >= 0.25:

        reasons.append(
            "Some Reed-Solomon codeword structure detected"
        )

    # --------------------------------------------------------
    # LDPC
    # --------------------------------------------------------

    ldpc_rate = float(
        ldpc.get(
            "syndrome_zero_rate",
            0.0
        )
    )

    if ldpc_rate >= 0.95:

        reasons.append(
            "Strong LDPC parity-check consistency"
        )

    elif ldpc_rate >= 0.80:

        reasons.append(
            "Moderate LDPC parity-check consistency"
        )

    # --------------------------------------------------------
    # Concatenated
    # --------------------------------------------------------

    if concatenated.get(
        "status"
    ) == "VALIDATED":

        reasons.append(
            "Convolutional and Reed-Solomon "
            "decoding validated together"
        )

    # --------------------------------------------------------
    # Existing statistical evidence
    # --------------------------------------------------------

    if float(
        parity.get(
            "parity_consistency",
            0.0
        )
    ) >= 0.75:

        reasons.append(
            "Parity/check-bit consistency observed"
        )

    if float(
        repetition.get(
            "repetition_score",
            0.0
        )
    ) >= 0.70:

        reasons.append(
            "Repeated-bit or repeated-block structure observed"
        )

    if float(
        hamming.get(
            "syndrome_evidence",
            0.0
        )
    ) >= 0.50:

        reasons.append(
            "Hamming-like parity structure observed"
        )

    if float(
        block.get(
            "best_correlation",
            0.0
        )
    ) >= 0.70:

        reasons.append(
            "Strong block correlation observed"
        )

    return reasons


# ============================================================
# FAMILY CANDIDATES
# ============================================================

def _family_candidates(
    conv,
    rs,
    concatenated,
    ldpc
):

    conv_score = float(
        conv.get(
            "evidence",
            0.0
        )
    )

    rs_score = float(
        rs.get(
            "evidence",
            0.0
        )
    )

    ldpc_score = float(
        ldpc.get(
            "evidence",
            0.0
        )
    )

    if concatenated.get(
        "status"
    ) == "VALIDATED":

        concat_score = 1.0

    elif concatenated.get(
        "status"
    ) == "CANDIDATE_ONLY":

        concat_score = min(
            0.75,
            max(
                conv_score,
                rs_score
            )
        )

    else:

        concat_score = (
            conv_score
            *
            rs_score
        )

    candidates = [
        {
            "family":
                "Convolutional + Viterbi",
            "evidence":
                round(
                    conv_score,
                    4
                ),
            "validated":
                conv_score >= 0.80,
        },
        {
            "family":
                "Reed-Solomon",
            "evidence":
                round(
                    rs_score,
                    4
                ),
            "validated":
                rs_score >= 0.75
                and
                rs.get(
                    "valid_codewords",
                    0
                ) > 0,
        },
        {
            "family":
                "Concatenated",
            "evidence":
                round(
                    float(
                        concat_score
                    ),
                    4
                ),
            "validated":
                concatenated.get(
                    "status"
                ) == "VALIDATED",
        },
        {
            "family":
                "LDPC",
            "evidence":
                round(
                    ldpc_score,
                    4
                ),
            "validated":
                ldpc_score >= 0.80,
        },
    ]

    candidates.sort(
        key=lambda item:
            item["evidence"],
        reverse=True
    )

    return candidates


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def _classify_fec(
    legacy_score,
    conv,
    rs,
    concatenated,
    ldpc,
    reasons
):

    conv_strong = (
        float(
            conv.get(
                "evidence",
                0.0
            )
        )
        >=
        0.80
    )

    rs_strong = (
        float(
            rs.get(
                "evidence",
                0.0
            )
        )
        >=
        0.75
    )

    ldpc_strong = (
        float(
            ldpc.get(
                "evidence",
                0.0
            )
        )
        >=
        0.80
    )

    concatenated_valid = (
        concatenated.get(
            "status"
        )
        ==
        "VALIDATED"
    )

    decoder_validated = (
        conv_strong
        or
        (
            rs_strong
            and
            rs.get(
                "valid_codewords",
                0
            ) > 0
        )
        or
        ldpc_strong
        or
        concatenated_valid
    )

    strong_sources = sum(
        [
            conv_strong,
            rs_strong,
            ldpc_strong,
            concatenated_valid,
        ]
    )

    if concatenated_valid:

        evidence = 1.0
        classification = "LIKELY"

    elif strong_sources >= 2:

        evidence = 0.90
        classification = "LIKELY"

    elif decoder_validated:

        evidence = 0.85
        classification = "LIKELY"

    elif (
        legacy_score >= 0.30
        or
        float(
            conv.get(
                "evidence",
                0.0
            )
        ) >= 0.50
        or
        float(
            rs.get(
                "evidence",
                0.0
            )
        ) >= 0.50
        or
        float(
            ldpc.get(
                "evidence",
                0.0
            )
        ) >= 0.50
    ):

        evidence = max(
            0.30,
            legacy_score
        )

        classification = "POSSIBLE"

    else:

        evidence = 0.0
        classification = "NONE"

    return (
        classification,
        float(
            np.clip(
                evidence,
                0.0,
                1.0
            )
        ),
    )


# ============================================================
# PUBLIC API
# ============================================================

def analyze_fec(
    bits,
    ldpc_matrix=None
):
    """
    Analyze a recovered bitstream for FEC.

    The function preserves the original SPECTRA API:

        analyze_fec(bits)

    Optional:

        analyze_fec(
            bits,
            ldpc_matrix=H
        )

    where H is a supported LDPC parity-check matrix.
    """

    bits = _clean_bits(
        bits
    )

    # ========================================================
    # BASIC EVIDENCE
    # ========================================================

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

    # ========================================================
    # PS-REQUIRED FEC CANDIDATES
    # ========================================================

    convolutional = _convolutional_evidence(
        bits
    )

    reed_solomon = _reed_solomon_evidence(
        bits
    )

    ldpc = _ldpc_evidence(
        bits,
        ldpc_matrix
    )

    # ========================================================
    # DECODING ATTEMPTS
    # ========================================================

    viterbi_result = _try_viterbi_decode(
        bits,
        convolutional
    )

    rs_decode_result = _try_reed_solomon_decode(
        bits,
        reed_solomon
    )

    concatenated_result = _try_concatenated_decode(
        bits,
        convolutional,
        reed_solomon
    )

    ldpc_result = auto_ldpc_search(bits)

    if ldpc_matrix is not None:

        ldpc_result = _ldpc_bitflip_decode(
            bits,
            ldpc_matrix
        )

    # ========================================================
    # SUPPORTING EVIDENCE SCORE
    # ========================================================

    legacy_score = _legacy_evidence_score(
        parity,
        repetition,
        hamming,
        block
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    reasons = _make_reasons(
        parity,
        repetition,
        hamming,
        block,
        convolutional,
        reed_solomon,
        ldpc,
        concatenated_result
    )

    # ========================================================
    # FINAL EVIDENCE
    # ========================================================

    (
        classification,
        score
    ) = _classify_fec(
        legacy_score,
        convolutional,
        reed_solomon,
        concatenated_result,
        ldpc,
        reasons
    )

    # ========================================================
    # FAMILY CANDIDATES
    # ========================================================

    candidates = _family_candidates(
        convolutional,
        reed_solomon,
        concatenated_result,
        ldpc
    )

    # ========================================================
    # IDENTIFIED FAMILY
    # ========================================================

    identified_family = "UNKNOWN"

    if (
        concatenated_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        identified_family = (
            "Concatenated"
        )

    elif (
        viterbi_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        identified_family = (
            "Convolutional + Viterbi"
        )

    elif (
        rs_decode_result.get(
            "status"
        )
        ==
        "SUCCESS"
    ):

        identified_family = (
            "Reed-Solomon"
        )

    elif (
        ldpc_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        identified_family = (
            "LDPC"
        )

    # If no validated decoder exists, do not force a family.
    if classification == "NONE":

        identified_family = "UNKNOWN"

    # ========================================================
    # SELECT CORRECTED BITSTREAM
    # ========================================================

    corrected_bits = None
    decoder_family = "NONE"
    decoder_status = "NOT_AVAILABLE"
    decoder_details = {}

    if (
        concatenated_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrected_bits = concatenated_result[
            "corrected_bits"
        ]

        decoder_family = "Concatenated"
        decoder_status = "VALIDATED"
        decoder_details = concatenated_result.get(
            "details",
            {}
        )

    elif (
        viterbi_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrected_bits = viterbi_result[
            "corrected_bits"
        ]

        decoder_family = (
            "Convolutional + Viterbi"
        )

        decoder_status = "VALIDATED"
        decoder_details = viterbi_result.get(
            "details",
            {}
        )

    elif (
        rs_decode_result.get(
            "status"
        )
        ==
        "SUCCESS"
    ):

        corrected_bits = rs_decode_result[
            "corrected_bits"
        ]

        decoder_family = "Reed-Solomon"
        decoder_status = "SUCCESS"
        decoder_details = rs_decode_result.get(
            "details",
            {}
        )

    elif (
        ldpc_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrected_bits = ldpc_result[
            "corrected_bits"
        ]

        decoder_family = "LDPC"
        decoder_status = "VALIDATED"
        decoder_details = {
            "decoder":
                "LDPC bit-flipping",
            "corrections":
                ldpc_result.get(
                    "corrections",
                    0
                ),
        }

    # ========================================================
    # CORRECTION COUNT
    # ========================================================

    corrections = 0

    if (
        concatenated_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrections = 0

    elif (
        viterbi_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrections = int(
            viterbi_result.get(
                "corrections",
                0
            )
        )

    elif (
        rs_decode_result.get(
            "status"
        )
        ==
        "SUCCESS"
    ):

        corrections = int(
            rs_decode_result.get(
                "corrections",
                0
            )
        )

    elif (
        ldpc_result.get(
            "status"
        )
        ==
        "VALIDATED"
    ):

        corrections = int(
            ldpc_result.get(
                "corrections",
                0
            )
        )

    # Ensure a successfully decoded Reed-Solomon result is not overridden by the heuristic Viterbi candidate.
    if (
        rs_decode_result.get("status") == "SUCCESS"
        and
        concatenated_result.get("status") != "VALIDATED"
    ):
        identified_family = "Reed-Solomon"
        corrected_bits = rs_decode_result.get("corrected_bits")
        decoder_family = "Reed-Solomon"
        decoder_status = "SUCCESS"
        decoder_details = rs_decode_result.get("details", {})
        corrections = int(rs_decode_result.get("corrections", 0))

    # ========================================================
    # PUBLIC RESULT
    # ========================================================

    return {

        # ----------------------------------------------------
        # Existing SPECTRA fields
        # ----------------------------------------------------

        "available":
            bool(
                len(bits) > 0
            ),

        "bits_analyzed":
            int(
                len(bits)
            ),

        "zero_percentage":
            balance[
                "zero_percentage"
            ],

        "one_percentage":
            balance[
                "one_percentage"
            ],

        "parity_group_size":
            parity[
                "best_group_size"
            ],

        "parity_evidence":
            parity[
                "parity_consistency"
            ],

        "best_repeat":
            repetition[
                "best_repeat"
            ],

        "repetition_evidence":
            repetition[
                "repetition_score"
            ],

        "copy_agreement":
            repetition[
                "copy_agreement"
            ],

        "hamming_code":
            hamming[
                "code"
            ],

        "hamming_codeword_size":
            hamming[
                "codeword_size"
            ],

        "hamming_valid_codeword_rate":
            hamming[
                "valid_codeword_rate"
            ],

        "hamming_syndrome_evidence":
            hamming[
                "syndrome_evidence"
            ],

        "hamming_codeword_count":
            hamming[
                "codeword_count"
            ],

        "best_block_size":
            block[
                "best_block_size"
            ],

        "block_correlation":
            block[
                "best_correlation"
            ],

        "code_rate_candidates":
            length[
                "rate_candidates"
            ],

        "fec_evidence":
            classification,

        "confidence":
            round(
                score * 100.0,
                2
            ),

        "evidence_reasons":
            reasons,

        # ----------------------------------------------------
        # New PS FEC fields
        # ----------------------------------------------------

        "supported_fec_families":
            SUPPORTED_FEC_FAMILIES,

        "identified_fec_family":
            identified_family,

        "fec_family_candidates":
            candidates,

        "convolutional_viterbi":
            {
                "available":
                    convolutional[
                        "available"
                    ],
                "best":
                    convolutional[
                        "best"
                    ],
                "candidates":
                    convolutional[
                        "candidates"
                    ],
                "evidence":
                    convolutional[
                        "evidence"
                    ],
                "decoder_status":
                    viterbi_result[
                        "status"
                    ],
            },

        "reed_solomon":
            {
                "available":
                    reed_solomon[
                        "available"
                    ],
                "best_nsym":
                    reed_solomon[
                        "best_nsym"
                    ],
                "offset":
                    reed_solomon[
                        "best_offset"
                    ],
                "codeword_size":
                    reed_solomon[
                        "codeword_size"
                    ],
                "syndrome_zero_rate":
                    reed_solomon[
                        "syndrome_zero_rate"
                    ],
                "codewords_tested":
                    reed_solomon[
                        "codewords_tested"
                    ],
                "valid_codewords":
                    reed_solomon[
                        "valid_codewords"
                    ],
                "evidence":
                    reed_solomon[
                        "evidence"
                    ],
                "decoder_library_available":
                    reed_solomon[
                        "decoder_available"
                    ],
                "decoder_status":
                    rs_decode_result[
                        "status"
                    ],
            },

        "concatenated":
            {
                "status":
                    concatenated_result[
                        "status"
                    ],
                "details":
                    concatenated_result.get(
                        "details",
                        {}
                    ),
            },

        "ldpc":
            {
                "available":
                    ldpc[
                        "available"
                    ],
                "matrix_supplied":
                    ldpc[
                        "matrix_supplied"
                    ],
                "matrix_shape":
                    ldpc[
                        "matrix_shape"
                    ],
                "syndrome_zero_rate":
                    ldpc[
                        "syndrome_zero_rate"
                    ],
                "evidence":
                    ldpc[
                        "evidence"
                    ],
                "decoder_status":
                    ldpc_result[
                        "status"
                    ],
                "message":
                    ldpc[
                        "message"
                    ],
            },

        # ----------------------------------------------------
        # Corrected / decoded output
        # ----------------------------------------------------

        "decoder_family":
            decoder_family,

        "decoder_status":
            decoder_status,

        "decoder_details":
            decoder_details,

        "corrected_bitstream":
            corrected_bits,

        "corrections":
            corrections,
    }