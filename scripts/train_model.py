from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.ml.classifier import train
train(ROOT/'models'/'modulation_rf_v16_data50.joblib',300)
print('SPECTRA modulation model trained.')
