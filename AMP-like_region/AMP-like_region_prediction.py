#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMP-like region prediction pipeline (single-model version)

Workflow
--------
1. Read full-length protein sequences and generate 13-aa sliding windows.
2. Use one trained AMP prediction model to classify each window.
3. Treat pred_label == 1 as AMP-positive.
4. Merge overlapping positive windows.
5. Calculate the proportion of AMP-like regions in each protein.

Example
-------
Run with the default example dataset:

    python AMP_region_prediction.py \
        --checkpoint /path/to/LysMiner_Model.ckpt

Run with a custom input file:

    python AMP_region_prediction.py \
        --input /path/to/protein_sequences.csv \
        --checkpoint /path/to/LysMiner_Model.ckpt \
        --output_dir /path/to/output \
        --device 0

Input format
------------
The input CSV file should contain at least two columns:

    id,seq
    Protein_1,MKT...
    Protein_2,MAL...

The first column is treated as the protein ID and the second column
as the amino acid sequence.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

# Directory containing this script
PROJECT_DIR = Path(__file__).resolve().parent

# Model-related files distributed with the repository
MODEL_SCRIPT = PROJECT_DIR / "mpbert_classification.py"
CONFIG_PATH = PROJECT_DIR / "config_1024.yaml"
VOCAB_FILE = PROJECT_DIR / "vocab_v2.txt"


# ============================================================
# Default parameters
# ============================================================

DEFAULT_INPUT = PROJECT_DIR / "example" / "example_seq.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "example"

WINDOW_SIZE = 13
STEP_SIZE = 1

# pred_label == 1 is considered AMP-positive
POSITIVE_LABEL = 1


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Predict AMP-like regions in full-length protein sequences "
            "using a sliding-window strategy and a trained AMP classifier."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV containing protein IDs and sequences. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.ckpt).",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for output files. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU/device ID used for prediction. Default: 0.",
    )

    parser.add_argument(
        "--seq_name",
        type=str,
        default="LYS",
        help=(
            "Prefix used for the sliding-window output filename. "
            "Default: LYS."
        ),
    )

    return parser.parse_args()


# ============================================================
# File checks
# ============================================================

def check_required_files(input_file, checkpoint):
    """Check whether all required files are available."""

    required_files = {
        "Input file": input_file,
        "Model checkpoint": checkpoint,
        "Prediction script": MODEL_SCRIPT,
        "Configuration file": CONFIG_PATH,
        "Vocabulary file": VOCAB_FILE,
    }

    missing_files = []

    for description, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append(
                f"{description}: {file_path}"
            )

    if missing_files:
        message = (
            "The following required files were not found:\n\n"
            + "\n".join(missing_files)
        )
        raise FileNotFoundError(message)


# ============================================================
# Step 1. Generate sliding-window sequences
# ============================================================

def generate_sliding_windows(
    input_file,
    window_file,
    window_size,
    step_size,
):
    """Generate fixed-length sliding windows from protein sequences."""

    print("\n" + "=" * 70)
    print("STEP 1: Generate sliding-window sequences")
    print("=" * 70)

    df = pd.read_csv(input_file, header=0)

    if df.shape[1] < 2:
        raise ValueError(
            "The input CSV must contain at least two columns: "
            "protein ID and amino acid sequence."
        )

    # First column = protein ID
    # Second column = amino acid sequence
    ids = df.iloc[:, 0].astype(str).tolist()
    sequences = df.iloc[:, 1].astype(str).tolist()

    new_ids = []
    window_sequences = []

    skipped_short_sequences = 0

    for protein_id, sequence in zip(ids, sequences):

        sequence = sequence.strip()

        if len(sequence) < window_size:
            skipped_short_sequences += 1
            continue

        for i in range(
            0,
            len(sequence) - window_size + 1,
            step_size,
        ):
            # i is zero-based.
            # Coordinates written to IDs are one-based.
            start_position = i + 1

            new_ids.append(
                f"{protein_id}_{start_position}"
            )

            window_sequences.append(
                sequence[i:i + window_size]
            )

    window_df = pd.DataFrame(
        {
            "id": new_ids,
            "seq": window_sequences,
        }
    )

    window_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_df.to_csv(
        window_file,
        index=False,
    )

    print(f"Input proteins          : {len(df)}")
    print(f"Total windows           : {len(window_df)}")
    print(
        f"Sequences < {window_size} aa : "
        f"{skipped_short_sequences}"
    )
    print(f"Window file             : {window_file}")

    if len(window_df) == 0:
        raise ValueError(
            "No sliding windows were generated. "
            "Please check the input sequences and window size."
        )


# ============================================================
# Step 2. Run model prediction
# ============================================================

def run_single_model_prediction(
    checkpoint,
    window_file,
    predict_dir,
    predict_result_file,
    device_id,
):
    """Run AMP prediction using one trained model checkpoint."""

    print("\n" + "=" * 70)
    print("STEP 2: Predict AMP-like windows with one model")
    print("=" * 70)

    predict_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir = predict_dir / "logs"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(MODEL_SCRIPT),

        "--config_path",
        str(CONFIG_PATH),

        "--load_checkpoint_url",
        str(checkpoint),

        "--do_predict",
        "True",

        "--description",
        "classification",

        "--num_class",
        "2",

        "--device_id",
        str(device_id),

        "--vocab_file",
        str(VOCAB_FILE),

        "--data_url",
        str(window_file),

        "--output_url",
        ".",

        "--return_sequence",
        "False",

        "--return_csv",
        "True",
    ]

    print(f"Checkpoint      : {checkpoint}")
    print(f"Prediction dir  : {predict_dir}")
    print(f"Device ID       : {device_id}")
    print(f"Positive label  : {POSITIVE_LABEL}")

    log_file = log_dir / "log.log"
    sys_log_file = log_dir / "sys.log"

    try:
        with (
            open(log_file, "w") as stdout_f,
            open(sys_log_file, "w") as stderr_f,
        ):
            subprocess.run(
                command,
                cwd=predict_dir,
                stdout=stdout_f,
                stderr=stderr_f,
                check=True,
            )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "\nModel prediction failed.\n"
            f"Please check:\n"
            f"  {log_file}\n"
            f"  {sys_log_file}\n"
        ) from error

    if not predict_result_file.exists():
        raise FileNotFoundError(
            "\nPrediction finished, but the expected result file "
            "was not found:\n"
            f"{predict_result_file}\n\n"
            "Please check:\n"
            f"  {log_file}\n"
            f"  {sys_log_file}\n\n"
            "Also confirm the output filename generated by "
            "mpbert_classification.py."
        )

    print(f"Prediction file : {predict_result_file}")


# ============================================================
# Step 3. AMP-like region calculation
# ============================================================

def parse_id(id_str):
    """
    Parse a sliding-window ID.

    Example
    -------
    Protein_A_32 -> (Protein_A, 32)

    rsplit is used so protein IDs may themselves contain underscores.
    """

    protein_name, start_pos = str(id_str).rsplit("_", 1)

    return pd.Series(
        [
            protein_name,
            int(start_pos),
        ]
    )


def positions_to_regions(positions):
    """
    Convert individual residue positions into continuous regions.

    Example
    -------
    {1, 2, 3, 4, 10, 11, 12}
        ->
    "1-4;10-12"
    """

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
            if start == previous:
                regions.append(str(start))
            else:
                regions.append(
                    f"{start}-{previous}"
                )

            start = pos
            previous = pos

    if start == previous:
        regions.append(str(start))
    else:
        regions.append(
            f"{start}-{previous}"
        )

    return ";".join(regions)


def calculate_amp_regions(
    predict_result_file,
    final_output_file,
    window_size,
):
    """
    Merge overlapping AMP-positive windows and calculate
    AMP-like region coverage for each protein.
    """

    print("\n" + "=" * 70)
    print(
        "STEP 3: Merge positive windows and "
        "calculate AMP-like regions"
    )
    print("=" * 70)

    df = pd.read_csv(
        predict_result_file
    )

    # Remove accidental CSV index columns.
    df = df.loc[
        :,
        ~df.columns.str.contains(r"^Unnamed"),
    ].copy()

    required_columns = {
        "id",
        "pred_label",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction result is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Current columns: {list(df.columns)}"
        )

    # Convert prediction labels to integers.
    df["pred_label"] = pd.to_numeric(
        df["pred_label"],
        errors="raise",
    ).astype(int)

    # Recover protein name and sliding-window start position.
    df[
        [
            "protein_name",
            "start_pos",
        ]
    ] = df["id"].apply(parse_id)

    results = []

    for protein_name, group in df.groupby(
        "protein_name",
        sort=False,
    ):

        max_start = group[
            "start_pos"
        ].max()

        protein_length = int(
            max_start + window_size - 1
        )

        # pred_label == 1 is AMP-positive.
        positive_group = group[
            group["pred_label"] == POSITIVE_LABEL
        ]

        amp_positions = set()

        for start in positive_group[
            "start_pos"
        ]:

            start = int(start)

            amp_positions.update(
                range(
                    start,
                    start + window_size,
                )
            )

        # Keep positions within protein boundaries.
        amp_positions = {
            pos
            for pos in amp_positions
            if 1 <= pos <= protein_length
        }

        amp_length = len(
            amp_positions
        )

        if protein_length > 0:
            amp_ratio = (
                amp_length / protein_length
            )
        else:
            amp_ratio = 0.0

        amp_regions = positions_to_regions(
            amp_positions
        )

        results.append(
            {
                "protein_name": protein_name,
                "protein_length": protein_length,
                "total_windows": len(group),
                "positive_windows": len(
                    positive_group
                ),
                "AMP_length": amp_length,
                "AMP_ratio": amp_ratio,
                "AMP_percent": amp_ratio * 100,
                "AMP_regions": amp_regions,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "AMP_ratio",
        ascending=False,
    )

    final_output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        final_output_file,
        index=False,
    )

    print("\nPrediction results:")
    print(result_df)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    proteins_with_amp = (
        result_df["AMP_length"] > 0
    )

    print(
        f"Positive definition          : "
        f"pred_label == {POSITIVE_LABEL}"
    )

    print(
        f"Total proteins               : "
        f"{len(result_df)}"
    )

    print(
        f"Proteins with AMP-like region: "
        f"{proteins_with_amp.sum()}"
    )

    print(
        "Proteins with AMP-like region (%): "
        f"{proteins_with_amp.mean() * 100:.2f}%"
    )

    print(
        "Mean AMP-like coverage       : "
        f"{result_df['AMP_percent'].mean():.2f}%"
    )

    print(
        "Median AMP-like coverage     : "
        f"{result_df['AMP_percent'].median():.2f}%"
    )

    print(
        "Maximum AMP-like coverage    : "
        f"{result_df['AMP_percent'].max():.2f}%"
    )

    print(
        "Minimum AMP-like coverage    : "
        f"{result_df['AMP_percent'].min():.2f}%"
    )

    print(
        f"\nFinal result saved to: "
        f"{final_output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    # Convert user-provided paths to absolute paths.
    input_file = args.input.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Output filenames
    window_file = output_dir / (
        f"{args.seq_name}_sliding_window_sequences"
        f"{WINDOW_SIZE}AA.csv"
    )

    predict_dir = (
        output_dir / "AMP_prediction"
    )

    predict_result_file = predict_dir / (
        f"{args.seq_name}_sliding_window_sequences"
        f"{WINDOW_SIZE}AA_predict_result.csv"
    )

    final_output_file = (
        output_dir /
        "protein_AMP_region_ratio.csv"
    )

    # Check required files.
    check_required_files(
        input_file=input_file,
        checkpoint=checkpoint,
    )

    print("\n" + "=" * 70)
    print("AMP-like region prediction pipeline")
    print("=" * 70)

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Input file        : {input_file}")
    print(f"Checkpoint        : {checkpoint}")
    print(f"Output directory  : {output_dir}")
    print(f"Window size       : {WINDOW_SIZE}")
    print(f"Step size         : {STEP_SIZE}")
    print(f"Device ID         : {args.device}")

    # Step 1
    generate_sliding_windows(
        input_file=input_file,
        window_file=window_file,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    # Step 2
    run_single_model_prediction(
        checkpoint=checkpoint,
        window_file=window_file,
        predict_dir=predict_dir,
        predict_result_file=predict_result_file,
        device_id=args.device,
    )

    # Step 3
    calculate_amp_regions(
        predict_result_file=predict_result_file,
        final_output_file=final_output_file,
        window_size=WINDOW_SIZE,
    )

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
