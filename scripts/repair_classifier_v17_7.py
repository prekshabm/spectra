from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = ROOT / "backend" / "ml" / "classifier.py"

# ------------------------------------------------------------
# Backup current file
# ------------------------------------------------------------

backup = ROOT / "backend" / "ml" / "classifier_before_v17_7_repair.py"
backup.write_text(
    CLASSIFIER.read_text(encoding="utf-8"),
    encoding="utf-8"
)

text = CLASSIFIER.read_text(
    encoding="utf-8"
).expandtabs(4)


# ------------------------------------------------------------
# Add V17.7 feature extractor import
# ------------------------------------------------------------

import_line = (
    "from backend.ml.fsk_specialist "
    "import extract_fsk_features"
)

if import_line not in text:

    anchor = (
        "from sklearn.ensemble "
        "import RandomForestClassifier"
    )

    text = text.replace(
        anchor,
        anchor + "\n" + import_line,
        1
    )


# ------------------------------------------------------------
# Remove duplicate V17.7 imports if any
# ------------------------------------------------------------

lines = text.splitlines()

clean = []
seen_import = False

for line in lines:

    if line.strip() == import_line:

        if seen_import:
            continue

        seen_import = True

    clean.append(line)

text = "\n".join(clean) + "\n"


# ------------------------------------------------------------
# Replace FSK specialist loader
# ------------------------------------------------------------

loader_start = "# V17.1 FSK SPECIALIST"
loader_end = "# LOAD MAIN MODEL"

start = text.find(loader_start)
end = text.find(loader_end, start)

if start == -1 or end == -1:

    raise RuntimeError(
        "Could not find the existing FSK specialist loader block."
    )

new_loader = """# V17.7 REAL-DATA FSK SPECIALIST

        self.fsk_specialist = None

        fsk_specialist_path = (
            self.path.parent
            / "specialist_fsk_real.joblib"
        )

        if fsk_specialist_path.exists():

            try:

                fsk_obj = joblib.load(
                    fsk_specialist_path
                )

                self.fsk_specialist = (
                    fsk_obj["model"]
                )

            except Exception:

                self.fsk_specialist = None

        """

text = (
    text[:start]
    + new_loader
    + text[end:]
)


# ------------------------------------------------------------
# Replace FSK prediction block
# ------------------------------------------------------------

prediction_start = "# V17.1 2-FSK / 4-FSK SPECIALIST"
prediction_end = "# Normalize"

start = text.find(prediction_start)
end = text.find(prediction_end, start)

if start == -1 or end == -1:

    raise RuntimeError(
        "Could not find the existing FSK prediction block."
    )

new_prediction = """# V17.7 REAL-DATA 2-FSK / 4-FSK SPECIALIST

        top_general = max(
            combined,
            key=combined.get
        )

        if (
            self.fsk_specialist is not None
            and top_general in (
                "2-FSK",
                "4-FSK"
            )
        ):

            try:

                fsk_x = extract_fsk_features(
                    signal,
                    fs
                )

                fsk_x = np.nan_to_num(
                    fsk_x,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6
                )

                fsk_prob = (
                    self.fsk_specialist
                    .predict_proba(
                        fsk_x.reshape(
                            1,
                            -1
                        )
                    )[0]
                )

                fsk_classes = list(
                    self.fsk_specialist.classes_
                )

                fsk_scores = dict(
                    zip(
                        fsk_classes,
                        fsk_prob
                    )
                )

                pair_total = (
                    combined.get(
                        "2-FSK",
                        0.0
                    )
                    +
                    combined.get(
                        "4-FSK",
                        0.0
                    )
                )

                combined["2-FSK"] = (
                    pair_total
                    * fsk_scores.get(
                        "2-FSK",
                        0.0
                    )
                )

                combined["4-FSK"] = (
                    pair_total
                    * fsk_scores.get(
                        "4-FSK",
                        0.0
                    )
                )

            except Exception:

                pass

        """

text = (
    text[:start]
    + new_prediction
    + text[end:]
)


# ------------------------------------------------------------
# Final whitespace cleanup
# ------------------------------------------------------------

text = text.expandtabs(4)

CLASSIFIER.write_text(
    text,
    encoding="utf-8"
)

print("V17.7 classifier repair completed.")
print(f"Backup created: {backup}")
print(f"Updated: {CLASSIFIER}")