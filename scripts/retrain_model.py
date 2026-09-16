from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.ml.classifier import train
print('Training robust SPECTRA modulation classifier...')
train(ROOT/'models'/'modulation_rf.joblib', n_per_class=300)
print('Done: models/modulation_rf.joblib')
