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

        bitstream_analysis = {
            "available": False,
            "reason": "No recovered bitstream available."
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

                if bitstream:

                    # --------------------------------------------
                    # BITSTREAM CORRELATION
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
                    # INTERLEAVING ANALYSIS
                    # --------------------------------------------

                    interleaving_result = analyze_interleaving(
                        bitstream
                    )

                    interleaving_result["available"] = True


                    # --------------------------------------------
                    # DEINTERLEAVING
                    # --------------------------------------------

                    try:

                        deinterleaving_result = (
                            apply_deinterleaver(
                                bitstream=bitstream,
                                mode=deinterleave_mode,
                                rows=deinterleave_rows,
                                branches=deinterleave_branches,
                                delay=deinterleave_delay,
                                seed=deinterleave_seed
                            )
                        )

                    except Exception as deinterleave_error:

                        deinterleaving_result = {
                            "available": True,
                            "success": False,
                            "type": str(
                                deinterleave_mode
                                or "none"
                            ).upper(),
                            "reason": str(
                                deinterleave_error
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