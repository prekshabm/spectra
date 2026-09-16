import numpy as np


# ============================================================
# VALIDATION
# ============================================================

def _validate_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) == 0:
        return bits.copy()

    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bits must contain only 0 and 1.")

    return bits


# ============================================================
# BLOCK INTERLEAVER
# ============================================================

def block_interleave(bits, rows=8):
    """
    Write bits row-wise and read column-wise.
    """

    bits = _validate_bits(bits)

    rows = int(rows)

    if rows < 2:
        raise ValueError("rows must be at least 2.")

    if len(bits) == 0:
        return bits.copy()

    if len(bits) % rows != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by "
            f"rows ({rows})."
        )

    cols = len(bits) // rows

    matrix = bits.reshape(rows, cols)

    return matrix.T.reshape(-1).astype(np.uint8)


def block_deinterleave(bits, rows=8):
    """
    Inverse of block_interleave().
    """

    bits = _validate_bits(bits)

    rows = int(rows)

    if rows < 2:
        raise ValueError("rows must be at least 2.")

    if len(bits) == 0:
        return bits.copy()

    if len(bits) % rows != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by "
            f"rows ({rows})."
        )

    cols = len(bits) // rows

    matrix = bits.reshape(cols, rows)

    return matrix.T.reshape(-1).astype(np.uint8)


# ============================================================
# CONVOLUTIONAL INTERLEAVER
# ============================================================

def convolutional_interleave(bits, branches=4, delay=3):
    """
    Deterministic convolutional interleaver.

    Successive bits are assigned cyclically to branches.
    Branch i has delay i * delay.
    """

    bits = _validate_bits(bits)

    branches = int(branches)
    delay = int(delay)

    if branches < 2:
        raise ValueError("branches must be at least 2.")

    if delay < 0:
        raise ValueError("delay must be non-negative.")

    if len(bits) == 0:
        return bits.copy()

    queues = [
        [0] * (i * delay)
        for i in range(branches)
    ]

    output = []

    branch = 0

    for bit in bits:

        queues[branch].append(int(bit))

        output.append(
            queues[branch].pop(0)
        )

        branch = (
            branch + 1
        ) % branches

    while any(queues):

        for branch in range(branches):

            if queues[branch]:

                output.append(
                    queues[branch].pop(0)
                )

    return np.asarray(
        output,
        dtype=np.uint8
    )


def convolutional_deinterleave(bits, branches=4, delay=3):
    """
    Exact inverse of convolutional_interleave().
    """

    bits = _validate_bits(bits)

    branches = int(branches)
    delay = int(delay)

    if branches < 2:
        raise ValueError("branches must be at least 2.")

    if delay < 0:
        raise ValueError("delay must be non-negative.")

    if len(bits) == 0:
        return bits.copy()

    flush_length = (
        delay
        * branches
        * (branches - 1)
        // 2
    )

    if len(bits) < flush_length:
        raise ValueError(
            "Bitstream is shorter than the convolutional "
            "interleaver flush length."
        )

    original_length = (
        len(bits) - flush_length
    )

    queues = [
        [None] * (i * delay)
        for i in range(branches)
    ]

    mapping = []

    branch = 0

    for index in range(original_length):

        queues[branch].append(index)

        mapping.append(
            queues[branch].pop(0)
        )

        branch = (
            branch + 1
        ) % branches

    while any(queues):

        for branch in range(branches):

            if queues[branch]:

                mapping.append(
                    queues[branch].pop(0)
                )

    if len(mapping) != len(bits):
        raise RuntimeError(
            "Convolutional mapping length mismatch."
        )

    recovered = np.zeros(
        original_length,
        dtype=np.uint8
    )

    for output_index, original_index in enumerate(mapping):

        if original_index is not None:

            recovered[original_index] = bits[
                output_index
            ]

    return recovered


# ============================================================
# DIAGONAL INTERLEAVER
# ============================================================

def _diagonal_positions(rows, cols):
    """
    Generate matrix positions in deterministic diagonal order.
    """

    positions = []

    for diagonal in range(
        rows + cols - 1
    ):

        for row in range(rows):

            col = diagonal - row

            if 0 <= col < cols:

                positions.append(
                    (row, col)
                )

    return positions


def diagonal_interleave(bits, rows=8):
    """
    Write row-wise and read along diagonals.
    """

    bits = _validate_bits(bits)

    rows = int(rows)

    if rows < 2:
        raise ValueError("rows must be at least 2.")

    if len(bits) == 0:
        return bits.copy()

    if len(bits) % rows != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by "
            f"rows ({rows})."
        )

    cols = len(bits) // rows

    matrix = bits.reshape(
        rows,
        cols
    )

    positions = _diagonal_positions(
        rows,
        cols
    )

    return np.asarray(
        [
            matrix[row, col]
            for row, col in positions
        ],
        dtype=np.uint8
    )


def diagonal_deinterleave(bits, rows=8):
    """
    Exact inverse of diagonal_interleave().
    """

    bits = _validate_bits(bits)

    rows = int(rows)

    if rows < 2:
        raise ValueError("rows must be at least 2.")

    if len(bits) == 0:
        return bits.copy()

    if len(bits) % rows != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by "
            f"rows ({rows})."
        )

    cols = len(bits) // rows

    matrix = np.zeros(
        (rows, cols),
        dtype=np.uint8
    )

    positions = _diagonal_positions(
        rows,
        cols
    )

    for value, (row, col) in zip(
        bits,
        positions
    ):
        matrix[row, col] = value

    return matrix.reshape(-1).astype(
        np.uint8
    )


# ============================================================
# PSEUDO-RANDOM INTERLEAVER
# ============================================================

def pseudo_random_interleave(bits, seed=42):
    """
    Deterministic pseudo-random interleaver.

    A seeded random permutation is applied to the bit positions.

    The seed is part of the interleaver configuration and must be
    known by the de-interleaver.
    """

    bits = _validate_bits(bits)

    if len(bits) == 0:
        return bits.copy()

    seed = int(seed)

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(
        len(bits)
    )

    return bits[
        permutation
    ].astype(np.uint8)


def pseudo_random_deinterleave(bits, seed=42):
    """
    Exact inverse of pseudo_random_interleave().
    """

    bits = _validate_bits(bits)

    if len(bits) == 0:
        return bits.copy()

    seed = int(seed)

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(
        len(bits)
    )

    recovered = np.empty_like(bits)

    recovered[
        permutation
    ] = bits

    return recovered.astype(
        np.uint8
    )


# ============================================================
# STRING HELPERS
# ============================================================

def bits_from_string(bitstream):
    """
    Convert '010101...' into uint8 bits.
    """

    if not isinstance(bitstream, str):
        raise TypeError(
            "bitstream must be a string."
        )

    cleaned = [
        int(b)
        for b in bitstream
        if b in ("0", "1")
    ]

    return np.asarray(
        cleaned,
        dtype=np.uint8
    )


def bits_to_string(bits):
    """
    Convert uint8 bits into '010101...' string.
    """

    bits = _validate_bits(bits)

    return "".join(
        bits.astype(str)
    )