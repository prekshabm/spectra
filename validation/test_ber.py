import numpy as np

from backend.demod.demodulator import demodulate


def generate_bpsk_signal(
    num_bits=10000,
    samples_per_symbol=8
):
    """Generate known BPSK IQ samples and original bits."""

    rng = np.random.default_rng(42)

    bits = rng.integers(
        0,
        2,
        num_bits
    ).astype(np.uint8)

    # SPECTRA BPSK convention:
    # 0 -> +1
    # 1 -> -1
    symbols = (
        1 - 2 * bits
    ).astype(np.float32)

    iq = np.repeat(
        symbols,
        samples_per_symbol
    ).astype(np.complex64)

    return bits, iq


def add_awgn(
    signal,
    snr_db,
    rng
):
    """Add complex AWGN at the requested SNR."""

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    noise_power = (
        signal_power /
        (10 ** (snr_db / 10))
    )

    noise = (
        rng.normal(
            0,
            np.sqrt(noise_power / 2),
            len(signal)
        )
        +
        1j * rng.normal(
            0,
            np.sqrt(noise_power / 2),
            len(signal)
        )
    )

    return (
        signal + noise
    ).astype(np.complex64)


def calculate_ber(
    reference_bits,
    recovered_bits
):
    """Calculate aligned BER."""

    n = min(
        len(reference_bits),
        len(recovered_bits)
    )

    errors = np.sum(
        reference_bits[:n] !=
        recovered_bits[:n]
    )

    return {
        "bit_errors": int(errors),
        "bits_compared": int(n),
        "ber": (
            float(errors / n)
            if n > 0
            else None
        )
    }


def main():

    sample_rate = 1_000_000
    symbol_rate = 125_000

    reference_bits, clean_iq = (
        generate_bpsk_signal()
    )

    rng = np.random.default_rng(123)

    print("AWGN BER validation")
    print("============================")

    print(
        "Reference bits:",
        len(reference_bits)
    )

    print(
        "Sample rate:",
        sample_rate
    )

    print(
        "Symbol rate:",
        symbol_rate
    )

    print(
        "Samples/symbol:",
        sample_rate / symbol_rate
    )

    print("\nSNR vs BER")
    print("----------------------------")

    for snr_db in [
        20,
        15,
        10,
        5,
        0
    ]:

        noisy_iq = add_awgn(
            clean_iq,
            snr_db,
            rng
        )

        result = demodulate(
            noisy_iq,
            "BPSK",
            sample_rate,
            symbol_rate
        )

        recovered_bits = np.array(
            [
                int(b)
                for b in result["bitstream"]
                if b in ("0", "1")
            ],
            dtype=np.uint8
        )

        ber = calculate_ber(
            reference_bits,
            recovered_bits
        )

        print(
            f"SNR: {snr_db:2d} dB | "
            f"Recovered: {len(recovered_bits):5d} | "
            f"Errors: {ber['bit_errors']:5d} | "
            f"BER: {ber['ber']:.6f}"
        )


if __name__ == "__main__":
    main()