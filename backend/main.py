from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np

from backend.signal_io.loader import load_signal
from backend.preprocessing.pipeline import preprocess
from backend.dsp.analysis import analyze_signal
from backend.ml.classifier import ModulationClassifier
from backend.demod.demodulator import demodulate
from backend.analysis.fec import analyze_fec
from backend.analysis.interleaving import analyze_interleaving
from backend.analysis.bitstream import analyze_bitstream

from backend.analysis.interleavers import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudo_random_deinterleave
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    ROOT
    / "models"
    / "modulation_rf.joblib"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SPECTRA v16",
    version="16.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


classifier = ModulationClassifier(
    MODEL
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "version": "16.0",
        "model_loaded": classifier.loaded
    }


# ============================================================
# DEINTERLEAVING
# ============================================================

def apply_deinterleaver(
    bitstream,
    mode,
    rows=8,
    branches=4,
    delay=3,
    seed=42
):
    """
    Apply the user-selected deinterleaver to a recovered
    binary bitstream.

    Supported modes:
        none
        block
        convolutional
        diagonal
        pseudo-random
    """

    mode = str(
        mode or "none"
    ).strip().lower()

    if mode == "none":

        return {
            "available": False,
            "success": False,
            "type": "NONE",
            "reason": "Deinterleaving not requested."
        }


    bits = np.asarray(
        [
            int(b)
            for b in str(bitstream)
            if b in ("0", "1")
        ],
        dtype=np.uint8
    )


    if len(bits) == 0:

        return {
            "available": True,
            "success": False,
            "type": mode.upper(),
            "reason": "No binary bits available for deinterleaving."
        }


    # --------------------------------------------------------
    # BLOCK
    # --------------------------------------------------------

    if mode == "block":

        rows = int(rows)

        if rows < 1:
            raise ValueError(
                "Block deinterleaver rows must be at least 1."
            )

        recovered = block_deinterleave(
            bits,
            rows=rows
        )

        return {
            "available": True,
            "success": True,
            "type": "BLOCK",
            "rows": rows,
            "input_bits": int(len(bits)),
            "output_bits": int(len(recovered)),
            "bitstream": "".join(
                str(int(b))
                for b in recovered
            )
        }


    # --------------------------------------------------------
    # CONVOLUTIONAL
    # --------------------------------------------------------

    if mode == "convolutional":

        branches = int(branches)
        delay = int(delay)

        if branches < 1:
            raise ValueError(
                "Convolutional deinterleaver branches must be at least 1."
            )

        if delay < 0:
            raise ValueError(
                "Convolutional deinterleaver delay cannot be negative."
            )

        recovered = convolutional_deinterleave(
            bits,
            branches=branches,
            delay=delay
        )

        return {
            "available": True,
            "success": True,
            "type": "CONVOLUTIONAL",
            "branches": branches,
            "delay": delay,
            "input_bits": int(len(bits)),
            "output_bits": int(len(recovered)),
            "bitstream": "".join(
                str(int(b))
                for b in recovered
            )
        }


    # --------------------------------------------------------
    # DIAGONAL
    # --------------------------------------------------------

    if mode == "diagonal":

        rows = int(rows)

        if rows < 1:
            raise ValueError(
                "Diagonal deinterleaver rows must be at least 1."
            )

        recovered = diagonal_deinterleave(
            bits,
            rows=rows
        )

        return {
            "available": True,
            "success": True,
            "type": "DIAGONAL",
            "rows": rows,
            "input_bits": int(len(bits)),
            "output_bits": int(len(recovered)),
            "bitstream": "".join(
                str(int(b))
                for b in recovered
            )
        }


    # --------------------------------------------------------
    # PSEUDO-RANDOM
    # --------------------------------------------------------

    if mode == "pseudo-random":

        seed = int(seed)

        recovered = pseudo_random_deinterleave(
            bits,
            seed=seed
        )

        return {
            "available": True,
            "success": True,
            "type": "PSEUDO-RANDOM",
            "seed": seed,
            "input_bits": int(len(bits)),
            "output_bits": int(len(recovered)),
            "bitstream": "".join(
                str(int(b))
                for b in recovered
            )
        }


    raise ValueError(
        f"Unsupported deinterleaving mode: {mode}"
    )

# ============================================================
# AUTOMATIC INTERLEAVING DETECTION
# ============================================================

def _bit_array(bitstream):
    return np.asarray(
        [int(b) for b in str(bitstream) if b in ("0", "1")],
        dtype=np.uint8
    )


def _transition_score(bits):
    if len(bits) < 2:
        return 0.0

    transitions = np.sum(bits[1:] != bits[:-1])
    return float(transitions / (len(bits) - 1))


def _periodicity_score(bits, max_lag=64):
    if len(bits) < 16:
        return 0.0, None

    x = (2.0 * bits.astype(np.float64)) - 1.0
    x = x - np.mean(x)

    energy = np.sum(x * x)

    if energy <= 0:
        return 0.0, None

    best_score = 0.0
    best_lag = None

    upper = min(max_lag, len(x) // 2)

    for lag in range(1, upper + 1):
        a = x[:-lag]
        b = x[lag:]

        denom = np.sqrt(
            np.sum(a * a) *
            np.sum(b * b)
        )

        if denom <= 0:
            continue

        score = abs(float(np.sum(a * b) / denom))

        if score > best_score:
            best_score = score
            best_lag = lag

    return best_score, best_lag


def _structure_score(bitstream):
    """
    Heuristic score used to compare candidate
    de-interleaved bitstreams.

    This is evidence-based and does not prove
    the true interleaving structure.
    """

    bits = _bit_array(bitstream)

    if len(bits) < 32:
        return 0.0

    periodicity, _ = _periodicity_score(bits)
    transition = _transition_score(bits)

    ones_ratio = float(np.mean(bits))

    balance = 1.0 - abs(ones_ratio - 0.5) * 2.0
    balance = max(0.0, min(1.0, balance))

    score = (
        0.60 * periodicity +
        0.20 * balance +
        0.20 * (1.0 - abs(transition - 0.5) * 2.0)
    )

    return float(max(0.0, min(1.0, score)))

# ============================================================
# AUTOMATIC INTERLEAVING DETECTION
# ============================================================

def _bit_array(bitstream):
    return np.asarray(
        [
            int(b)
            for b in str(bitstream)
            if b in ("0", "1")
        ],
        dtype=np.uint8
    )


def _transition_score(bits):
    if len(bits) < 2:
        return 0.0

    return float(
        np.mean(bits[1:] != bits[:-1])
    )


def _autocorrelation_score(bits, max_lag=64):
    """
    Measures periodic structure in a bitstream.
    """

    if len(bits) < 32:
        return 0.0

    x = (
        2.0 * bits.astype(np.float64)
    ) - 1.0

    x -= np.mean(x)

    energy = np.sum(x * x)

    if energy <= 0:
        return 0.0

    best = 0.0

    upper = min(
        max_lag,
        len(x) // 2
    )

    for lag in range(1, upper + 1):

        a = x[:-lag]
        b = x[lag:]

        denom = np.sqrt(
            np.sum(a * a) *
            np.sum(b * b)
        )

        if denom <= 0:
            continue

        corr = abs(
            float(
                np.sum(a * b) / denom
            )
        )

        best = max(
            best,
            corr
        )

    return float(best)


def _structure_score(bitstream):
    """
    Estimate how much deterministic structure exists
    in a candidate recovered bitstream.

    This is a heuristic metric and does not by itself
    prove the interleaving type.
    """

    bits = _bit_array(bitstream)

    if len(bits) < 32:
        return 0.0

    transition = _transition_score(
        bits
    )

    periodicity = _autocorrelation_score(
        bits
    )

    ones_ratio = float(
        np.mean(bits)
    )

    balance = (
        1.0
        - abs(ones_ratio - 0.5) * 2.0
    )

    balance = max(
        0.0,
        min(1.0, balance)
    )

    transition_structure = (
        1.0
        - abs(transition - 0.5) * 2.0
    )

    score = (
        0.50 * periodicity
        + 0.25 * balance
        + 0.25 * transition_structure
    )

    return float(
        max(
            0.0,
            min(1.0, score)
        )
    )


def detect_interleaving_type(
    bitstream,
    rows=8,
    branches=4,
    delay=3,
    seed=42,
    min_confidence=0.60
):
    """
    Evidence-based automatic interleaving detector.

    Supported hypotheses:
        BLOCK
        CONVOLUTIONAL
        DIAGONAL
        PSEUDO-RANDOM

    The detector tests each supported de-interleaver
    and compares the resulting candidate bitstreams.

    It abstains when the evidence is insufficient.
    """

    bits = _bit_array(
        bitstream
    )

    input_length = len(bits)

    if input_length < 32:

        return {
            "available": True,
            "detected": False,
            "type": "NOT_CONFIDENTLY_IDENTIFIED",
            "confidence": 0.0,
            "score": None,
            "margin": None,
            "reason": (
                "Insufficient recovered bits for "
                "interleaving hypothesis testing."
            ),
            "candidates": []
        }


    # --------------------------------------------------------
    # TEST EACH HYPOTHESIS
    # --------------------------------------------------------

    hypotheses = [
        {
            "mode": "block",
            "name": "BLOCK",
            "rows": rows,
            "branches": branches,
            "delay": delay,
            "seed": seed
        },

        {
            "mode": "convolutional",
            "name": "CONVOLUTIONAL",
            "rows": rows,
            "branches": branches,
            "delay": delay,
            "seed": seed
        },

        {
            "mode": "diagonal",
            "name": "DIAGONAL",
            "rows": rows,
            "branches": branches,
            "delay": delay,
            "seed": seed
        },

        {
            "mode": "pseudo-random",
            "name": "PSEUDO-RANDOM",
            "rows": rows,
            "branches": branches,
            "delay": delay,
            "seed": seed
        }
    ]


    candidates = []


    for hypothesis in hypotheses:

        mode = hypothesis["mode"]

        try:

            # ------------------------------------------------
            # LENGTH COMPATIBILITY
            # ------------------------------------------------

            compatible = True

            if mode in (
                "block",
                "diagonal"
            ):

                if input_length % rows != 0:
                    compatible = False

            elif mode == "convolutional":

                flush_length = (
                    delay
                    * branches
                    * (branches - 1)
                    // 2
                )

                if input_length < flush_length:
                    compatible = False

            elif mode == "pseudo-random":

                compatible = (
                    input_length > 0
                )


            if not compatible:

                candidates.append({
                    "type": hypothesis["name"],
                    "score": 0.0,
                    "structural_score": 0.0,
                    "length_compatible": False,
                    "parameters": {}
                })

                continue


            # ------------------------------------------------
            # APPLY DEINTERLEAVER
            # ------------------------------------------------

            result = apply_deinterleaver(
                bitstream=bitstream,
                mode=mode,
                rows=rows,
                branches=branches,
                delay=delay,
                seed=seed
            )


            if not result.get(
                "success",
                False
            ):

                candidates.append({
                    "type": hypothesis["name"],
                    "score": 0.0,
                    "structural_score": 0.0,
                    "length_compatible": True,
                    "parameters": {},
                    "reason": result.get(
                        "reason",
                        "Deinterleaving failed."
                    )
                })

                continue


            candidate_bits = result.get(
                "bitstream",
                ""
            )


            # ------------------------------------------------
            # STRUCTURE SCORE
            # ------------------------------------------------

            structure = _structure_score(
                candidate_bits
            )


            # ------------------------------------------------
            # LENGTH SCORE
            # ------------------------------------------------

            output_length = len(
                _bit_array(
                    candidate_bits
                )
            )

            if mode == "convolutional":

                expected_output_length = (
                    input_length
                    - (
                        delay
                        * branches
                        * (branches - 1)
                        // 2
                    )
                )

                length_score = (
                    1.0
                    if output_length
                    == expected_output_length
                    else 0.0
                )

            else:

                length_score = (
                    1.0
                    if output_length
                    == input_length
                    else 0.0
                )


            # ------------------------------------------------
            # FINAL CANDIDATE SCORE
            # ------------------------------------------------

            score = (
                0.80 * structure
                + 0.20 * length_score
            )


            candidates.append({
                "type": hypothesis["name"],
                "score": round(
                    float(score),
                    4
                ),
                "structural_score": round(
                    float(structure),
                    4
                ),
                "length_compatible": True,
                "input_bits": input_length,
                "output_bits": output_length,
                "parameters": {
                    "rows": rows,
                    "branches": branches,
                    "delay": delay,
                    "seed": seed
                },
                "bitstream": candidate_bits
            })


        except Exception as error:

            candidates.append({
                "type": hypothesis["name"],
                "score": 0.0,
                "structural_score": 0.0,
                "length_compatible": False,
                "parameters": {},
                "reason": str(error)
            })


    # --------------------------------------------------------
    # NO VALID CANDIDATES
    # --------------------------------------------------------

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.get(
            "length_compatible",
            False
        )
        and "bitstream" in candidate
    ]


    if not valid_candidates:

        return {
            "available": True,
            "detected": False,
            "type": "NOT_CONFIDENTLY_IDENTIFIED",
            "confidence": 0.0,
            "score": None,
            "margin": None,
            "reason": (
                "No valid interleaving hypothesis "
                "could be evaluated."
            ),
            "candidates": candidates
        }


    # --------------------------------------------------------
    # RANK CANDIDATES
    # --------------------------------------------------------

    valid_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    best = valid_candidates[0]

    best_score = float(
        best["score"]
    )


    second_score = (
        float(
            valid_candidates[1]["score"]
        )
        if len(valid_candidates) > 1
        else 0.0
    )


    margin = (
        best_score
        - second_score
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = min(
        1.0,
        max(
            0.0,
            0.70 * best_score
            + 0.30 * margin
        )
    )


    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    detected = (
        best_score >= min_confidence
        and margin >= 0.05
    )


    if detected:

        detected_type = best["type"]

        reason = (
            "Best-supported interleaving hypothesis "
            "selected from the supported structures."
        )

        selected_bitstream = best[
            "bitstream"
        ]

        parameters = best[
            "parameters"
        ]

    else:

        detected_type = (
            "NOT_CONFIDENTLY_IDENTIFIED"
        )

        reason = (
            "The available evidence is insufficient "
            "to confidently identify the interleaving "
            "structure."
        )

        selected_bitstream = ""

        parameters = {}


    # --------------------------------------------------------
    # CLEAN CANDIDATE OUTPUT
    # --------------------------------------------------------

    candidate_summary = []

    for candidate in candidates:

        candidate_summary.append({
            "type": candidate["type"],
            "score": candidate.get(
                "score",
                0.0
            ),
            "structural_score": candidate.get(
                "structural_score",
                0.0
            ),
            "length_compatible": candidate.get(
                "length_compatible",
                False
            ),
            "input_bits": candidate.get(
                "input_bits"
            ),
            "output_bits": candidate.get(
                "output_bits"
            ),
            "parameters": candidate.get(
                "parameters",
                {}
            )
        })


    return {
        "available": True,
        "detected": detected,
        "type": detected_type,
        "confidence": round(
            float(confidence),
            4
        ),
        "score": round(
            best_score,
            4
        ),
        "margin": round(
            float(margin),
            4
        ),
        "reason": reason,
        "candidates": candidate_summary,
        "selected_bitstream": selected_bitstream,
        "parameters": parameters
    }
# ============================================================
# ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    sample_rate: float = Form(1_000_000),
    dtype: str = Form("float32"),
    iq_format: str = Form("IQ"),

    deinterleave_mode: str = Form("none"),

    deinterleave_rows: int = Form(8),

    deinterleave_branches: int = Form(4),

    deinterleave_delay: int = Form(3),

    deinterleave_seed: int = Form(42)
):

    raw = await file.read()

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        sig, fs, meta = load_signal(
            file.filename,
            raw,
            sample_rate,
            dtype,
            iq_format
        )

        # Keep original signal for physical measurements.
        raw_sig = sig.copy()

        # ----------------------------------------------------
        # DSP ANALYSIS
        # ----------------------------------------------------

        result = analyze_signal(
            raw_sig,
            fs
        )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        sig, prep = preprocess(
            raw_sig
        )

        # ----------------------------------------------------
        # ML CLASSIFICATION
        # ----------------------------------------------------

        features = result["features"]

        cls = classifier.predict(
            features,
            sig,
            fs
        )

        # Internal ML features are not exposed.
        result.pop(
            "features",
            None
        )

        # ----------------------------------------------------
        # DEMODULATION
        # ----------------------------------------------------

        modulation = cls.get(
            "detected"
        )

        demod_result = {
            "available": False,
            "reason": "No modulation classification available."
        }

        fec_result = {
            "available": False,
            "reason": "No recovered bitstream available."
        }

        interleaving_result = {
            "available": False,
            "reason": "No recovered bitstream available."
        }

        deinterleaving_result = {
            "available": False,
            "success": False,
            "type": "NONE",
            "reason": "Deinterleaving not requested."
        }


        if modulation:

            try:

                symbol_rate = (
                    result
                    .get("parameters", {})
                    .get("symbol_rate_candidate_sym_s")
                )

                demod_result = demodulate(
                    sig,
                    modulation,
                    fs,
                    symbol_rate
                )

                demod_result["available"] = True

                # ------------------------------------------------
                # RECOVERED BITSTREAM
                # ------------------------------------------------

                bitstream = demod_result.get(
                    "bitstream",
                    ""
                )

                bitstream_analysis = {
                    "available": False,
                    "reason": "No recovered bitstream available."
                }

                if bitstream:

                    # --------------------------------------------
                    # BITSTREAM STRUCTURE ANALYSIS
                    # --------------------------------------------

                    bitstream_analysis = analyze_bitstream(
                        bitstream
                    )

                    bitstream_analysis["available"] = True


                    # --------------------------------------------
                    # FEC
                    # --------------------------------------------

                    fec_result = analyze_fec(
                        bitstream
                    )

                    fec_result["available"] = True


                    # --------------------------------------------
                    # AUTOMATIC INTERLEAVING DETECTION
                    # --------------------------------------------

                    interleaving_result = detect_interleaving_type(
                        bitstream=bitstream,
                        rows=deinterleave_rows,
                        branches=deinterleave_branches,
                        delay=deinterleave_delay,
                        seed=deinterleave_seed
                    )


                    # --------------------------------------------
                    # AUTOMATIC DEINTERLEAVING
                    # --------------------------------------------

                    if interleaving_result.get("detected"):

                        detected_mode = (
                            interleaving_result["type"]
                            .strip()
                            .lower()
                        )

                        detected_parameters = (
                            interleaving_result.get(
                                "parameters",
                                {}
                            )
                        )

                        try:

                            deinterleaving_result = (
                                apply_deinterleaver(
                                    bitstream=bitstream,
                                    mode=detected_mode,
                                    rows=detected_parameters.get(
                                        "rows",
                                        deinterleave_rows
                                    ),
                                    branches=detected_parameters.get(
                                        "branches",
                                        deinterleave_branches
                                    ),
                                    delay=detected_parameters.get(
                                        "delay",
                                        deinterleave_delay
                                    ),
                                    seed=detected_parameters.get(
                                        "seed",
                                        deinterleave_seed
                                    )
                                )
                            )

                            deinterleaving_result["automatic"] = True

                        except Exception as deinterleave_error:

                            deinterleaving_result = {
                                "available": True,
                                "success": False,
                                "automatic": True,
                                "type": detected_mode.upper(),
                                "reason": str(
                                    deinterleave_error
                                )
                            }

                    else:

                        deinterleaving_result = {
                            "available": True,
                            "success": False,
                            "automatic": True,
                            "type": "NONE",
                            "reason": (
                                "Interleaving type was not "
                                "confidently identified; "
                                "de-interleaving was not applied."
                            )
                        }

            except Exception as demod_error:

                demod_result = {
                    "available": False,
                    "reason": str(demod_error)
                }

                fec_result = {
                    "available": False,
                    "reason": "Demodulation failed."
                }

                interleaving_result = {
                    "available": False,
                    "reason": "Demodulation failed."
                }

                deinterleaving_result = {
                    "available": False,
                    "success": False,
                    "type": "NONE",
                    "reason": "Demodulation failed."
                }


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        result.update({

            "classification": cls,

            "demodulation": demod_result,

            "bitstream_analysis": bitstream_analysis,

            "fec": fec_result,

            "interleaving": interleaving_result,

            "deinterleaving": deinterleaving_result,

            "preprocessing": prep,

            "source": meta

        })

        return result


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(
            ROOT / "frontend"
        ),
        html=True
    ),
    name="frontend"
)