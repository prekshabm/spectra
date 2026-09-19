import numpy as np


def estimate_frequency_offset(
    signal,
    fs,
    modulation=None
):
    """
    Estimate carrier frequency offset using
    modulation-dependent power transformation.
    """

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    if len(x) < 32:
        return 0.0

    modulation = str(
        modulation or ""
    ).strip().upper()

    if modulation == "BPSK":
        order = 2

    elif modulation == "QPSK":
        order = 4

    else:
        order = 2

    z = x ** order

    phase_diff = np.angle(
        z[1:] * np.conj(z[:-1])
    )

    transformed_offset = (
        fs
        * np.mean(phase_diff)
        / (2.0 * np.pi)
    )

    frequency_offset = (
        transformed_offset / order
    )

    return float(
        frequency_offset
    )



def correct_frequency_offset(
    signal,
    fs,
    frequency_offset
):
    """
    Correct carrier frequency offset.
    """

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    if len(x) == 0:
        return x

    n = np.arange(
        len(x),
        dtype=np.float64
    )

    correction = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequency_offset
        * n
        / fs
    )

    return (
        x * correction
    ).astype(np.complex64)


def synchronize_signal(
    signal,
    fs,
    modulation=None
):
    """
    Estimate and correct frequency offset.
    """

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    frequency_offset = (
        estimate_frequency_offset(
            x,
            fs,
            modulation
        )
    )

    corrected = (
        correct_frequency_offset(
            x,
            fs,
            frequency_offset
        )
    )

    return {
        "signal": corrected,
        "frequency_offset_hz": frequency_offset
    }
