#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMP-like region prediction pipeline (single-fold version)

Workflow:
1. Read full-length protein sequences and generate 13-aa sliding windows.
2. Run ONE trained model (fold_0 checkpoint) to predict each window.
3. Treat pred_label == 1 as AMP-positive, merge overlapping positive windows,
   and calculate AMP-like region coverage for each protein.

This script combines, in order:
    sliding_windows.py
    00-predict_ABP-MPB.sh
    04-cal-quyu.py
"""

import os
import subprocess
from pathlib import Path
import pandas as pd


# ============================================================

# 0. Parameters

# ============================================================

# Project root directory

PROJECT_DIR = Path("/data4/gh/LysMiner-GITHUB/AMP-like_region")

# Example/input directory

WORK_DIR = PROJECT_DIR / "example"

# Input protein sequence file

INPUT_FILE = WORK_DIR / "example_seq.csv"

# Sliding-window parameters

SEQ_NAME = "LYS"

WINDOW_SIZE = 13

STEP_SIZE = 1

# Sliding-window output

WINDOW_FILE = WORK_DIR / (

    f"{SEQ_NAME}_sliding_window_sequences{WINDOW_SIZE}AA.csv"

)

# Prediction output directory

PREDICT_DIR = WORK_DIR / "AMP_prediction"

# ============================================================

# Model files

# ============================================================

# AMP prediction model checkpoint

CHECKPOINT = Path(

    "/data2/gaohan/UniDL4Biopep/UniDL4Biopep_earlystop_fold10_x3/"

    "f16/fold_0/saved_models_sel/test_Best_Model.ckpt"

)

# All model-related files are located in PROJECT_DIR

MODEL_SCRIPT = PROJECT_DIR / "mpbert_classification.py"

CONFIG_PATH = PROJECT_DIR / "config_1024.yaml"

VOCAB_FILE = PROJECT_DIR / "vocab_v2.txt"

# ============================================================

# Prediction parameters

# ============================================================

DEVICE_ID = 6

# label == 1 is AMP-positive

POSITIVE_LABEL = 1

# ============================================================

# Output files

# ============================================================

PREDICT_RESULT_FILE = PREDICT_DIR / (

    f"{SEQ_NAME}_sliding_window_sequences"

    f"{WINDOW_SIZE}AA_predict_result.csv"

)

FINAL_OUTPUT_FILE = WORK_DIR / "protein_AMP_region_ratio.csv"


# ============================================================
# 1. Generate sliding-window sequences
# ============================================================

def generate_sliding_windows():
    print("\n" + "=" * 70)
    print("STEP 1: Generate sliding-window sequences")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, header=0)

    # Same convention as the original script:
    # column 1 = protein ID, column 2 = sequence.
    ids = df.iloc[:, 0].astype(str).tolist()
    sequences = df.iloc[:, 1].astype(str).tolist()

    new_ids = []
    seqs = []

    for protein_id, sequence in zip(ids, sequences):
        sequence = sequence.strip()

        for i in range(0, len(sequence) - WINDOW_SIZE + 1, STEP_SIZE):
            # i is zero-based; sequence coordinates written to ID are one-based.
            new_ids.append(f"{protein_id}_{i + 1}")
            seqs.append(sequence[i:i + WINDOW_SIZE])

    window_df = pd.DataFrame({"id": new_ids, "seq": seqs})
    window_df.to_csv(WINDOW_FILE, index=False)

    print(f"Input proteins : {len(df)}")
    print(f"Total windows  : {len(window_df)}")
    print(f"Window file    : {WINDOW_FILE}")


# ============================================================
# 2. Run a single model prediction
# ============================================================

def run_single_fold_prediction():
    print("\n" + "=" * 70)
    print("STEP 2: Predict AMP-like windows with ONE model")
    print("=" * 70)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = PREDICT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        str(MODEL_SCRIPT),
        "--config_path", str(CONFIG_PATH),
        "--load_checkpoint_url", str(CHECKPOINT),
        "--do_predict", "True",
        "--description", "classification",
        "--num_class", "2",
        "--device_id", str(DEVICE_ID),
        "--vocab_file", str(VOCAB_FILE),
        "--data_url", str(WINDOW_FILE),
        "--output_url", ".",
        "--return_sequence", "False",
        "--return_csv", "True",
    ]

    print(f"Checkpoint      : {CHECKPOINT}")
    print(f"Prediction dir  : {PREDICT_DIR}")
    print(f"Positive label  : {POSITIVE_LABEL}")

    log_file = log_dir / "log.log"
    sys_log_file = log_dir / "sys.log"

    with open(log_file, "w") as stdout_f, open(sys_log_file, "w") as stderr_f:
        subprocess.run(
            command,
            cwd=PREDICT_DIR,
            stdout=stdout_f,
            stderr=stderr_f,
            check=True,
        )

    if not PREDICT_RESULT_FILE.exists():
        raise FileNotFoundError(
            "Prediction finished, but the expected result file was not found:\n"
            f"{PREDICT_RESULT_FILE}\n"
            "Please check logs/log.log and logs/sys.log, and confirm the filename "
            "generated by mpbert_classification_1x_v2.py."
        )

    print(f"Prediction file : {PREDICT_RESULT_FILE}")


# ============================================================
# 3. Merge positive windows and calculate AMP-like coverage
# ============================================================

def parse_id(id_str):
    """Example: LYS3_CRAVI_32 -> (LYS3_CRAVI, 32)."""
    protein_name, start_pos = str(id_str).rsplit("_", 1)
    return pd.Series([protein_name, int(start_pos)])


def positions_to_regions(positions):
    """Convert {1,2,3,4,10,11,12} to '1-4;10-12'."""
    if not positions:
        return ""

    positions = sorted(positions)
    regions = []

    start = positions[0]
    previous = positions[0]

    for pos in positions[1:]:
        if pos == previous + 1:
            previous = pos
        else:
            regions.append(str(start) if start == previous else f"{start}-{previous}")
            start = pos
            previous = pos

    regions.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(regions)


def calculate_amp_regions():
    print("\n" + "=" * 70)
    print("STEP 3: Merge positive windows and calculate AMP-like regions")
    print("=" * 70)

    df = pd.read_csv(PREDICT_RESULT_FILE)

    # Remove accidental CSV index columns.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    required_columns = {"id", "pred_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction result is missing required columns: {sorted(missing)}\n"
            f"Current columns: {list(df.columns)}"
        )

    # Make comparison robust if pred_label was read as text/float.
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="raise").astype(int)

    df[["protein_name", "start_pos"]] = df["id"].apply(parse_id)

    results = []

    for protein_name, group in df.groupby("protein_name", sort=False):
        max_start = group["start_pos"].max()
        protein_length = int(max_start + WINDOW_SIZE - 1)

        # label == 1 is AMP-positive.
        positive_group = group[group["pred_label"] == POSITIVE_LABEL]

        amp_positions = set()
        for start in positive_group["start_pos"]:
            amp_positions.update(range(int(start), int(start) + WINDOW_SIZE))

        amp_positions = {
            pos for pos in amp_positions
            if 1 <= pos <= protein_length
        }

        amp_length = len(amp_positions)
        amp_ratio = amp_length / protein_length if protein_length > 0 else 0.0
        amp_regions = positions_to_regions(amp_positions)

        results.append({
            "protein_name": protein_name,
            "protein_length": protein_length,
            "total_windows": len(group),
            "positive_windows": len(positive_group),
            "AMP_length": amp_length,
            "AMP_ratio": amp_ratio,
            "AMP_percent": amp_ratio * 100,
            "AMP_regions": amp_regions,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("AMP_ratio", ascending=False)
    result_df.to_csv(FINAL_OUTPUT_FILE, index=False)

    print(result_df)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Positive definition          : pred_label == {POSITIVE_LABEL}")
    print(f"Total proteins               : {len(result_df)}")
    print(f"Proteins with AMP-like region: {(result_df['AMP_length'] > 0).sum()}")
    print(
        "Proteins with AMP-like region (%): "
        f"{(result_df['AMP_length'] > 0).mean() * 100:.2f}%"
    )
    print(f"Mean AMP-like coverage       : {result_df['AMP_percent'].mean():.2f}%")
    print(f"Median AMP-like coverage     : {result_df['AMP_percent'].median():.2f}%")
    print(f"Maximum AMP-like coverage    : {result_df['AMP_percent'].max():.2f}%")
    print(f"Minimum AMP-like coverage    : {result_df['AMP_percent'].min():.2f}%")
    print(f"\nFinal result saved to: {FINAL_OUTPUT_FILE}")


# ============================================================
# Main
# ============================================================

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    generate_sliding_windows()
    run_single_fold_prediction()
    calculate_amp_regions()

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
