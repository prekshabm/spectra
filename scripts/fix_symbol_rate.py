from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "backend" / "dsp" / "analysis.py"


NEW_FUNCTION = r'''def _estimate_symbol_rate(x, fs):
    """
    Estimate a symbol-rate candidate from timing structure in the IQ signal.

    The previous estimator searched only a tiny lag range at common sample
    rates, which could return values such as 500 ksym/s for a 1 MHz signal.

    This version:
      - searches a realistic samples/symbol range,
      - uses signal-transition activity,
      - also uses instantaneous-frequency activity,
      - looks for autocorrelation peaks,
      - returns a candidate only when timing evidence is meaningful.

    The result remains a candidate estimate, not a guaranteed symbol rate.
    """

    x = np.asarray(x, dtype=np.complex128)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 256:
        return None

    fs = float(fs)

    if not np.isfinite(fs) or fs <= 0:
        return None

    # Remove DC and normalize.
    x = x - np.mean(x)

    rms = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    x = x / rms

    # ------------------------------------------------------------
    # Candidate samples/symbol range
    # ------------------------------------------------------------
    #
    # For 1 MHz signals this searches roughly 2..256 samples/symbol,
    # corresponding to about 3.9 ksym/s .. 500 ksym/s.
    #
    # This is deliberately broad enough for real recordings while
    # remaining computationally manageable.
    # ------------------------------------------------------------

    min_sps = 2
    max_sps = min(
        256,
        max(
            min_sps + 2,
            int(fs / 1_000.0)
        )
    )

    if max_sps <= min_sps:
        return None

    # ------------------------------------------------------------
    # Timing activity
    # ------------------------------------------------------------

    # Complex-sample transition activity.
    transition = np.abs(np.diff(x))

    # Amplitude-change activity helps with QAM.
    amplitude = np.abs(x)
    amplitude_activity = np.abs(np.diff(amplitude))

    # Instantaneous-frequency activity helps with FSK.
    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    freq_activity = np.abs(
        phase_diff - np.median(phase_diff)
    )

    sequences = [
        transition,
        amplitude_activity,
        freq_activity,
    ]

    # ------------------------------------------------------------
    # Downsample for speed
    # ------------------------------------------------------------

    max_points = 50000

    processed = []

    for seq in sequences:

        seq = np.asarray(
            seq,
            dtype=float
        )

        seq = np.nan_to_num(
            seq,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        seq = seq - np.mean(seq)

        step = max(
            1,
            len(seq) // max_points
        )

        seq = seq[::step]

        if len(seq) < 128:
            continue

        processed.append(
            (
                seq,
                step
            )
        )

    if not processed:
        return None

    # ------------------------------------------------------------
    # Score candidate lags
    # ------------------------------------------------------------

    scores = np.zeros(
        max_sps + 1,
        dtype=float
    )

    weights = np.zeros_like(scores)

    for seq, step in processed:

        # Limit lag to what exists in this downsampled sequence.
        max_lag = min(
            max_sps // step if step > 1 else max_sps,
            len(seq) // 4
        )

        if max_lag < min_sps:
            continue

        energy = float(
            np.dot(seq, seq)
        )

        if energy <= 1e-12:
            continue

        for sps in range(
            min_sps,
            max_sps + 1
        ):

            lag = int(
                round(sps / step)
            )

            if lag < 1 or lag >= max_lag:
                continue

            a = seq[:-lag]
            b = seq[lag:]

            denom = (
                np.sqrt(
                    np.dot(a, a)
                    * np.dot(b, b)
                )
                + 1e-12
            )

            corr = abs(
                float(np.dot(a, b))
            ) / denom

            if np.isfinite(corr):
                scores[sps] += corr
                weights[sps] += 1.0

    valid = weights > 0

    if not np.any(valid):
        return None

    scores[valid] /= weights[valid]

    # Ignore extremely weak timing evidence.
    valid_scores = scores.copy()
    valid_scores[~valid] = -np.inf

    # ------------------------------------------------------------
    # Prefer a fundamental timing period over a large harmonic.
    # ------------------------------------------------------------

    best_sps = None
    best_score = -np.inf

    for sps in range(
        min_sps,
        max_sps + 1
    ):

        if not np.isfinite(valid_scores[sps]):
            continue

        score = valid_scores[sps]

        # If a smaller divisor has nearly the same evidence,
        # prefer the smaller candidate.
        for divisor in (
            2,
            3,
            4,
        ):
            smaller = sps // divisor

            if (
                smaller >= min_sps
                and
                smaller <= max_sps
                and
                np.isfinite(valid_scores[smaller])
                and
                valid_scores[smaller] >= 0.90 * score
            ):
                score *= 0.85

        if score > best_score:
            best_score = score
            best_sps = sps

    if best_sps is None:
        return None

    # Require at least modest autocorrelation evidence.
    if best_score < 0.08:
        return None

    rate = fs / float(best_sps)

    if not (
        1_000.0
        <= rate
        <= fs / 2.0
    ):
        return None

    return float(rate)
'''


print("=" * 70)
print("SPECTRA SYMBOL-RATE ESTIMATOR V2")
print("=" * 70)

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

# Backup first.
backup = TARGET.with_suffix(
    TARGET.suffix + ".bak_symbolrate"
)

if not backup.exists():
    shutil.copy2(
        TARGET,
        backup
    )
    print()
    print("Backup created:")
    print(backup)
else:
    print()
    print("Backup already exists:")
    print(backup)

text = TARGET.read_text(
    encoding="utf-8"
)

pattern = re.compile(
    r"def _estimate_symbol_rate\(x, fs\):.*?(?=\ndef analyze_signal\(x, fs\):)",
    re.DOTALL,
)

match = pattern.search(text)

if not match:
    raise RuntimeError(
        "Could not locate _estimate_symbol_rate() "
        "in backend/dsp/analysis.py"
    )

updated = (
    text[:match.start()]
    + NEW_FUNCTION
    + "\n\n"
    + text[match.end():]
)

TARGET.write_text(
    updated,
    encoding="utf-8"
)

print()
print("Updated:")
print(TARGET)

print()
print("=" * 70)
print("TESTING NEW ESTIMATOR")
print("=" * 70)

test_code = r'''
from pathlib import Path

from backend.signal_io.loader import load_signal
from backend.dsp.analysis import analyze_signal


mods = {
    "BPSK": "data/bpsk",
    "QPSK": "data/qpsk",
    "2-FSK": "data/2fsk",
    "4-FSK": "data/4fsk",
    "16-QAM": "data/16qam",
}

for modulation, folder in mods.items():

    files = sorted(
        Path(folder).glob("*.iq")
    )

    if not files:
        print(f"{modulation:7s} -> NO FILE")
        continue

    path = files[0]

    signal, fs, meta = load_signal(
        path.name,
        path.read_bytes(),
        1_000_000,
        dtype="float32",
        iq_format="IQ",
    )

    result = analyze_signal(
        signal,
        fs
    )

    rate = result.get(
        "symbol_rate"
    )

    if rate is None:
        text = "None"
    else:
        text = f"{rate:.2f} Sym/s"

    print(
        f"{modulation:7s} -> {text}"
    )
'''

subprocess.run(
    [
        sys.executable,
        "-c",
        test_code,
    ],
    cwd=ROOT,
    check=True,
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)