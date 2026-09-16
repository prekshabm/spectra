import numpy as np
from reedsolo import RSCodec, ReedSolomonError


# ============================================================
# REED-SOLOMON FEC
#
# Uses the reedsolo library.
#
# Default:
#     nsym = 10 parity symbols
#
# Reed-Solomon works on BYTES/SYMBOLS rather than individual
# bits. For a message of k bytes and nsym parity bytes:
#
#     transmitted length = k + nsym
#
# The decoder can correct up to:
#
#     nsym / 2
#
# byte errors.
# ============================================================


DEFAULT_NSYM = 10


def _clean_bytes(data):
    """
    Convert input into a uint8 NumPy array.
    """

    if isinstance(data, bytes):
        return np.frombuffer(
            data,
            dtype=np.uint8
        ).copy()

    if isinstance(data, bytearray):
        return np.frombuffer(
            bytes(data),
            dtype=np.uint8
        ).copy()

    data = np.asarray(
        data,
        dtype=np.uint8
    )

    return data.copy()


# ============================================================
# ENCODER
# ============================================================

def reed_solomon_encode(
    data,
    nsym=DEFAULT_NSYM,
):
    """
    Encode data using Reed-Solomon FEC.

    Parameters
    ----------
    data:
        Input bytes or uint8 array.

    nsym:
        Number of parity symbols.

    Returns
    -------
    np.ndarray
        Encoded data containing message + parity.
    """

    data = _clean_bytes(data)

    nsym = int(nsym)

    if nsym <= 0:
        raise ValueError(
            "nsym must be greater than 0."
        )

    codec = RSCodec(nsym)

    encoded = codec.encode(
        bytes(data)
    )

    return np.frombuffer(
        encoded,
        dtype=np.uint8
    ).copy()


# ============================================================
# DECODER
# ============================================================

def reed_solomon_decode(
    received,
    nsym=DEFAULT_NSYM,
):
    """
    Decode a Reed-Solomon protected sequence.

    Returns only the recovered original message bytes.

    Raises ReedSolomonError when the corruption is beyond
    the correction capability.
    """

    received = _clean_bytes(
        received
    )

    nsym = int(nsym)

    if nsym <= 0:
        raise ValueError(
            "nsym must be greater than 0."
        )

    codec = RSCodec(nsym)

    decoded = codec.decode(
        bytes(received)
    )

    # reedsolo may return different structures depending
    # on version. The first item is the decoded message.
    if isinstance(decoded, tuple):
        decoded = decoded[0]

    return np.frombuffer(
        bytes(decoded),
        dtype=np.uint8
    ).copy()


# ============================================================
# CORRUPTION UTILITY
# ============================================================

def introduce_errors(
    data,
    positions,
    xor_values=None,
):
    """
    Deliberately corrupt selected byte positions.

    This is used for controlled testing.

    positions:
        List/array of byte indexes.

    xor_values:
        Optional list of XOR values.
        Default = 0xFF for every position.
    """

    data = _clean_bytes(
        data
    )

    positions = np.asarray(
        positions,
        dtype=np.int64
    )

    if np.any(
        (positions < 0)
        |
        (positions >= len(data))
    ):
        raise IndexError(
            "Error position is outside the data."
        )

    corrupted = data.copy()

    if xor_values is None:

        xor_values = np.full(
            len(positions),
            0xFF,
            dtype=np.uint8
        )

    else:

        xor_values = np.asarray(
            xor_values,
            dtype=np.uint8
        )

        if len(xor_values) != len(positions):
            raise ValueError(
                "xor_values and positions "
                "must have equal length."
            )

    for pos, value in zip(
        positions,
        xor_values
    ):
        corrupted[pos] ^= value

    return corrupted


# ============================================================
# BYTE ERROR COUNT
# ============================================================

def byte_errors(
    original,
    recovered,
):
    """
    Count byte positions that differ.
    """

    original = _clean_bytes(
        original
    )

    recovered = _clean_bytes(
        recovered
    )

    if len(original) != len(recovered):
        raise ValueError(
            "Sequences must have equal length."
        )

    return int(
        np.count_nonzero(
            original != recovered
        )
    )


# ============================================================
# CLASS WRAPPER
# ============================================================

class ReedSolomonFEC:

    def __init__(
        self,
        nsym=DEFAULT_NSYM,
    ):

        self.nsym = int(nsym)

        if self.nsym <= 0:
            raise ValueError(
                "nsym must be greater than 0."
            )

        self.codec = RSCodec(
            self.nsym
        )

    def encode(self, data):

        return reed_solomon_encode(
            data,
            self.nsym
        )

    def decode(self, received):

        return reed_solomon_decode(
            received,
            self.nsym
        )