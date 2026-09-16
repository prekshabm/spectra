from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
VALIDATION_DIR = ROOT / "validation_50"

FILES_PER_CLASS = 10
RANDOM_SEED = 42

CLASS_FOLDERS = {
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "2fsk": "2-FSK",
    "4fsk": "4-FSK",
    "16qam": "16-QAM",
}

random.seed(RANDOM_SEED)

print("=" * 70)
print("SPECTRA V16 — NEW 60/50/10 DATA SPLIT")
print("=" * 70)

print()
print("60 files/class")
print("50 training/class")
print("10 validation/class")
print()

if VALIDATION_DIR.exists():
    raise RuntimeError(
        f"{VALIDATION_DIR} already exists. "
        "Delete it before running this script again."
    )

VALIDATION_DIR.mkdir(parents=True)

total = 0

for folder_name, class_name in CLASS_FOLDERS.items():

    source = DATA_DIR / folder_name
    destination = VALIDATION_DIR / class_name

    if not source.exists():
        raise RuntimeError(f"Missing folder: {source}")

    files = sorted(source.glob("*.iq"))

    print("-" * 70)
    print(class_name)
    print(f"Found: {len(files)} files")

    if len(files) != 60:
        raise RuntimeError(
            f"{class_name}: expected exactly 60 files, "
            f"found {len(files)}"
        )

    selected = random.sample(files, FILES_PER_CLASS)

    destination.mkdir(parents=True)

    for file in selected:
        shutil.copy2(file, destination / file.name)

    print("Validation files:")
    for file in selected:
        print(f"  {file.name}")

    total += len(selected)

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

for folder in sorted(VALIDATION_DIR.iterdir()):
    count = len(list(folder.glob("*.iq")))
    print(f"{folder.name:<10} {count:>2} validation files")

print()
print(f"TOTAL VALIDATION FILES: {total}")

if total != 50:
    raise RuntimeError("Validation set is not exactly 50 files.")

print()
print("Training files remaining: 50 per class")
print("Validation files:         10 per class")
print()
print("Original data files were NOT modified.")
print("Existing models were NOT modified.")