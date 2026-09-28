"""Bounded, conservative LDPC candidate search for SPECTRA."""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _clean(bits: Iterable[int] | str) -> np.ndarray:
    if isinstance(bits, str):
        arr = np.fromiter(
            (1 if b == "1" else 0 for b in bits if b in "01"),
            dtype=np.uint8
        )
    else:
        arr = np.asarray(list(bits), dtype=np.uint8)
        arr = arr[(arr == 0) | (arr == 1)]
    return arr


def _make_h(n: int, rate: str, seed: int) -> np.ndarray | None:
    rates = {
        "1/2": 0.5,
        "2/3": 2 / 3,
        "3/4": 0.75,
        "5/6": 5 / 6,
        "7/8": 7 / 8,
    }

    if rate not in rates:
        return None

    m = int(round(n * (1.0 - rates[rate])))

    if m <= 0 or m >= n:
        return None

    if rate == "1/2":
        check_degree, var_degree = 6, 3
    elif rate == "2/3":
        check_degree, var_degree = 6, 2
    elif rate == "3/4":
        check_degree, var_degree = 8, 2
    elif rate == "5/6":
        check_degree, var_degree = 6, 1
    else:
        check_degree, var_degree = 8, 1

    rng = np.random.default_rng(seed)

    H = np.zeros((m, n), dtype=np.uint8)

    check_slots = np.repeat(np.arange(m), check_degree)
    var_slots = np.repeat(np.arange(n), var_degree)

    edges = min(len(check_slots), len(var_slots))

    rng.shuffle(check_slots)
    rng.shuffle(var_slots)

    for i in range(edges):
        H[check_slots[i], var_slots[i]] = 1

    for j in range(n):
        if not H[:, j].any():
            H[int(rng.integers(0, m)), j] = 1

    for i in range(m):
        if not H[i, :].any():
            H[i, int(rng.integers(0, n))] = 1

    return H


def _syndrome_rate(block: np.ndarray, H: np.ndarray) -> float:
    return float(np.mean((H @ block.astype(np.uint8)) & 1))


def _bitflip(block: np.ndarray, H: np.ndarray, max_iter: int = 8):
    x = block.copy()
    flips = 0

    for _ in range(max_iter):
        syn = (H @ x) & 1
        bad = np.flatnonzero(syn)

        if bad.size == 0:
            break

        votes = H[bad].sum(axis=0)
        threshold = max(1, int(np.ceil(0.55 * bad.size)))

        candidates = np.flatnonzero(votes >= threshold)

        if candidates.size == 0:
            break

        order = candidates[np.argsort(-votes[candidates])]
        selected = order[:min(8, max(1, len(order) // 8))]

        x[selected] ^= 1
        flips += int(selected.size)

    return x, flips


def auto_ldpc_search(
    bits: Iterable[int] | str,
    *,
    block_lengths=(96, 128, 192, 256, 384, 512, 648, 1024),
    rates=("1/2", "2/3", "3/4", "5/6", "7/8"),
    seeds=(7, 17, 37),
) -> dict:

    x = _clean(bits)

    result = {
        "decoder": "LDPC",
        "status": "SEARCHED",
        "validated": False,
        "evaluation": "AUTO_CANDIDATE_SEARCH",
        "candidates_tested": 0,
        "reason": "No supported LDPC candidate passed validation.",
        "block_length": None,
        "code_rate": None,
        "candidate_seed": None,
        "syndrome_rate_before": None,
        "syndrome_rate_after": None,
        "corrected_bits": 0,
    }

    if len(x) < 192:
        result.update({
            "status": "INSUFFICIENT_DATA",
            "reason": (
                "Need at least 192 recovered bits for "
                "bounded LDPC candidate search."
            ),
        })
        return result

    best = None

    for n in block_lengths:

        if n <= 0 or len(x) < n or len(x) % n:
            continue

        blocks = x.reshape(-1, n)

        if len(blocks) < 4:
            continue

        for rate in rates:

            for seed in seeds:

                H = _make_h(n, rate, seed)

                if H is None:
                    continue

                before = []
                after = []
                corrections = 0

                for block in blocks[:min(16, len(blocks))]:

                    b0 = _syndrome_rate(block, H)

                    decoded, flips = _bitflip(
                        block,
                        H
                    )

                    b1 = _syndrome_rate(
                        decoded,
                        H
                    )

                    before.append(b0)
                    after.append(b1)
                    corrections += flips

                b0 = float(np.mean(before))
                b1 = float(np.mean(after))
                improvement = b0 - b1

                candidate = (
                    b1,
                    -improvement
                )

                result["candidates_tested"] += 1

                if best is None or candidate < best[0]:
                    best = (
                        candidate,
                        n,
                        rate,
                        seed,
                        b0,
                        b1,
                        corrections,
                        len(before),
                    )

                if (
                    b0 <= 0.22
                    and b1 <= 0.05
                    and improvement >= 0.10
                ):

                    result.update({
                        "status": "VALIDATED",
                        "validated": True,
                        "reason": (
                            "Bounded LDPC candidate passed "
                            "multi-block syndrome and "
                            "bit-flip validation."
                        ),
                        "block_length": n,
                        "code_rate": rate,
                        "candidate_seed": seed,
                        "syndrome_rate_before": round(b0, 6),
                        "syndrome_rate_after": round(b1, 6),
                        "corrected_bits": corrections,
                        "blocks_used": len(before),
                        "decoder": "LDPC bit-flip",
                    })

                    return result

    if best is not None:

        (
            _,
            n,
            rate,
            seed,
            b0,
            b1,
            corrections,
            blocks_used,
        ) = best

        result.update({
            "reason": (
                "LDPC candidates were evaluated, but none "
                "passed the validation threshold for this bitstream."
            ),
            "block_length": n,
            "code_rate": rate,
            "candidate_seed": seed,
            "syndrome_rate_before": round(b0, 6),
            "syndrome_rate_after": round(b1, 6),
            "corrected_bits": corrections,
            "blocks_used": blocks_used,
        })

    return result
