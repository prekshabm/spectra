from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_model import (
    discover_files,
    load_v16_signal,
)


def frequency_features(signal, fs):

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    x = x[np.isfinite(x)]

    if len(x) < 256:
        return None

    # Normalize
    power = np.mean(
        np.abs(x) ** 2
    )

    x = x / (
        np.sqrt(power)
        + 1e-12
    )

    # Instantaneous phase difference
    dphi = np.angle(
        x[1:]
        * np.conj(x[:-1])
    )

    # Convert to Hz
    freq = (
        dphi
        * fs
        / (2 * np.pi)
    )

    # Ignore extreme outliers
    lo = np.percentile(
        freq,
        2
    )

    hi = np.percentile(
        freq,
        98
    )

    freq = freq[
        (freq >= lo)
        & (freq <= hi)
    ]

    if len(freq) < 100:
        return None

    # Smooth
    window = 9

    kernel = (
        np.ones(window)
        / window
    )

    smooth = np.convolve(
        freq,
        kernel,
        mode="same"
    )

    # Statistics
    mean_freq = np.mean(
        smooth
    )

    std_freq = np.std(
        smooth
    )

    median_freq = np.median(
        smooth
    )

    q10, q25, q50, q75, q90 = np.percentile(
        smooth,
        [10, 25, 50, 75, 90]
    )

    # Histogram
    bins = 80

    hist, edges = np.histogram(
        smooth,
        bins=bins,
        density=True
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2

    # Smooth histogram
    hs = np.convolve(
        hist,
        np.ones(5) / 5,
        mode="same"
    )

    # Significant histogram bins
    threshold = (
        np.max(hs)
        * 0.15
    )

    significant = (
        hs > threshold
    )

    # Number of contiguous regions
    transitions = np.diff(
        significant.astype(int)
    )

    regions = int(
        np.sum(
            transitions == 1
        )
    )

    if significant[0]:
        regions += 1

    # Spectral analysis
    n = min(
        len(x),
        32768
    )

    windowed = (
        x[:n]
        * np.hanning(n)
    )

    spectrum = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft(
                    windowed
                )
            )
        ) ** 2
    )

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            1 / fs
        )
    )

    spectrum /= (
        np.max(spectrum)
        + 1e-12
    )

    # Strong spectrum peaks
    peak_threshold = (
        np.max(spectrum)
        * 0.10
    )

    strong = (
        spectrum
        > peak_threshold
    )

    spec_transitions = np.diff(
        strong.astype(int)
    )

    spectral_regions = int(
        np.sum(
            spec_transitions == 1
        )
    )

    if strong[0]:
        spectral_regions += 1

    return {
        "mean_freq": mean_freq,
        "std_freq": std_freq,
        "median_freq": median_freq,
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "hist_regions": regions,
        "spectral_regions": spectral_regions,
    }


print("=" * 85)
print("SPECTRA V17.2 - FSK FEATURE ANALYSIS")
print("=" * 85)

files = discover_files()

fsk_files = [
    (path, label)
    for path, label in files
    if label in (
        "2-FSK",
        "4-FSK"
    )
]

print(
    f"\nFound {len(fsk_files)} FSK validation files.\n"
)

rows = []

for path, true_label in fsk_files:

    print("-" * 85)
    print(
        f"{path.name}"
    )

    signal, fs, _ = load_v16_signal(
        path
    )

    result = frequency_features(
        signal,
        fs
    )

    if result is None:
        print("Unable to analyze")
        continue

    row = {
        "label": true_label,
        **result
    }

    rows.append(
        row
    )

    print(
        f"True               : {true_label}"
    )

    print(
        f"Mean frequency     : "
        f"{result['mean_freq']:,.2f} Hz"
    )

    print(
        f"Frequency std      : "
        f"{result['std_freq']:,.2f} Hz"
    )

    print(
        f"Frequency median   : "
        f"{result['median_freq']:,.2f} Hz"
    )

    print(
        f"Q10 / Q90          : "
        f"{result['q10']:,.2f} / "
        f"{result['q90']:,.2f} Hz"
    )

    print(
        f"Histogram regions  : "
        f"{result['hist_regions']}"
    )

    print(
        f"Spectral regions   : "
        f"{result['spectral_regions']}"
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

print("\n")
print("=" * 85)
print("GROUP SUMMARY")
print("=" * 85)

for label in (
    "2-FSK",
    "4-FSK"
):

    group = [
        r
        for r in rows
        if r["label"] == label
    ]

    if not group:
        continue

    print(
        f"\n{label}"
    )

    print("-" * 85)

    for feature in (
        "std_freq",
        "hist_regions",
        "spectral_regions"
    ):

        values = np.array(
            [
                r[feature]
                for r in group
            ],
            dtype=float
        )

        print(
            f"{feature:20s}"
            f" mean={np.mean(values):.3f}"
            f"  median={np.median(values):.3f}"
            f"  min={np.min(values):.3f}"
            f"  max={np.max(values):.3f}"
        )