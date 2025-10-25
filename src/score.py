#!/usr/bin/env python3
"""
Prediction-only scoring script for Geyser dataset.

Reads:  data/raw/geyser.tsv
Model:  models/linear_regression_pipeline.joblib
Writes: data/scored/geyser_scored.csv (columns: eruptions, waiting [if present], predicted_waiting)

Note: plotting is intentionally excluded — plotting is done separately per request.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "linear_regression_pipeline.joblib"
INPUT_PATH = ROOT / "data" / "raw" / "geyser.tsv"
OUT_DIR = ROOT / "data" / "scored"
OUT_CSV = OUT_DIR / "geyser_scored.csv"

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    return df

def predict_and_save(model, df: pd.DataFrame, out_csv: Path):
    if "eruptions" not in df.columns:
        raise KeyError("Input dataframe must contain 'eruptions' column")

    X = df[["eruptions"]]
    preds = model.predict(X)

    # safety checks
    if not np.all(np.isfinite(preds)):
        raise ValueError("Predictions contain non-finite values")

    out_df = df.copy()
    out_df["predicted_waiting"] = preds

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote scored CSV to {out_csv}")

def main():
    try:
        model = load_model(MODEL_PATH)
        df = read_data(INPUT_PATH)

        if df.empty:
            print("Warning: input dataframe is empty.", file=sys.stderr)

        predict_and_save(model, df, OUT_CSV)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Scoring script for Geyser dataset.

Reads:  data/raw/geyser.tsv
Model:  models/linear_regression_pipeline.joblib
Writes: data/scored/geyser_scored.csv
       plots/geyser_predictions.png
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "models" / "linear_regression_pipeline.joblib"
INPUT_PATH = ROOT / "data" / "raw" / "geyser.tsv"
OUT_DIR = ROOT / "data" / "scored"
PLOTS_DIR = ROOT / "plots"
OUT_CSV = OUT_DIR / "geyser_scored.csv"
OUT_PLOT = PLOTS_DIR / "geyser_predictions.png"

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    # try TSV first, fallback to CSV
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    return df

def predict_and_save(model, df: pd.DataFrame, out_csv: Path, out_plot: Path):
    if "eruptions" not in df.columns:
        raise KeyError("Input dataframe must contain 'eruptions' column")

    X = df[["eruptions"]]
    preds = model.predict(X)

    # safety checks
    if not np.all(np.isfinite(preds)):
        raise ValueError("Predictions contain non-finite values")

    # attach predictions
    out_df = df.copy()
    out_df["predicted_waiting"] = preds

    # ensure output dirs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote scored CSV to {out_csv}")

# Plot actual vs predicted (using 'waiting' if present)
    plt.figure(figsize=(6, 4))
    if "waiting" in out_df.columns:
        plt.scatter(out_df["waiting"], out_df["predicted_waiting"], alpha=0.7)
        plt.xlabel("Actual waiting")
    else:
        plt.scatter(out_df["eruptions"], out_df["predicted_waiting"], alpha=0.7)
        plt.xlabel("Eruptions")
    plt.ylabel("Predicted waiting")
    plt.title("Geyser: actual vs predicted waiting")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Wrote plot to {out_plot}")


def main():
    try:
        model = load_model(MODEL_PATH)
        df = read_data(INPUT_PATH)

        if df.empty:
            print("Warning: input dataframe is empty.", file=sys.stderr)

        predict_and_save(model, df, OUT_CSV, OUT_PLOT)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()