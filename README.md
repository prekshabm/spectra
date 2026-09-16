# SPECTRA v16

Local FastAPI + NumPy/SciPy + scikit-learn RF signal-analysis prototype.

## VS Code / Windows

Open the `SPECTRA_v16` folder in VS Code, then in the terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\train_model.py
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Test file
Use your `QPSK_1000000Hz_10dB.iq`, set sample rate to `1000000`, dtype `float32`, and IQ format `IQ`.

## Pipeline
File parsing → preprocessing → FFT/PSD → bandwidth/SNR/parameter estimation → ML modulation classification → visualization.

The modulation classifier is trained on synthetic BPSK, QPSK, 2-FSK, 4-FSK and 16-QAM examples with randomized SNR, phase and frequency offset. It returns probabilities rather than a hard-coded label.


ivbowiR
