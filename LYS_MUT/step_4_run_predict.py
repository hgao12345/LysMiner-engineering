#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate lysozyme variants using a trained LYS-MUT model.

The script uses a target-specific mask_Best_Model.ckpt generated
during LYS-MUT training to generate candidate variants from a
wild-type lysozyme sequence.

Example
-------
python step_4_run_predict.py \
    --input ./example/example_seq.fasta \
    --model_dir ./example/example_seq/train_data \
    --mpbert_dir /path/to/MP-BERT-v3 \
    --vocab_file /path/to/vocab_v2.txt \
    --device 0

Default generation parameters
-----------------------------
Number of generation attempts : 100000
Masking proportion             : 0.1
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# 1. Parse arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate lysozyme variants using a trained "
            "LYS-MUT model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input FASTA file containing the wild-type lysozyme.",
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the trained "
            "mask_Best_Model.ckpt checkpoint."
        ),
    )

    parser.add_argument(
        "--mpbert_dir",
        type=Path,
        required=True,
        help=(
            "Path to the MP-BERT directory containing "
            "mpbert_mask.py and config_1024.yaml."
        ),
    )

    parser.add_argument(
        "--vocab_file",
        type=Path,
        required=True,
        help="Path to vocab_v2.txt.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Ascend device ID used for prediction (default: 0).",
    )

    parser.add_argument(
        "--predict_mask_num",
        type=int,
        default=100000,
        help=(
            "Number of sequence-generation attempts "
            "(default: 100000)."
        ),
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.1,
        help=(
            "Masking proportion used for sequence generation "
            "(default: 0.1)."
        ),
    )

    return parser.parse_args()


# ============================================================
# 2. Check required files
# ============================================================

def check_required_files(
    input_file,
    model_path,
    mpbert_script,
    config_file,
    vocab_file,
):
    """Check all files required for prediction."""

    required_files = [
        input_file,
        model_path,
        mpbert_script,
        config_file,
        vocab_file,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following required files were not found:\n"
            f"{missing_text}"
        )


# ============================================================
# 3. Run prediction
# ============================================================

def run_prediction(
    input_file,
    model_path,
    save_path,
    mpbert_script,
    config_file,
    vocab_file,
    device,
    predict_mask_num,
    mask_prob,
):
    """Generate candidate variants using the trained LYS-MUT model."""

    command = [
        sys.executable,
        str(mpbert_script),

        "--config_path",
        str(config_file),

        "--vocab_file",
        str(vocab_file),

        "--do_predict",
        "True",

        "--description",
        "sequence",

        "--device_id",
        str(device),

        "--data_url",
        str(input_file),

        "--load_checkpoint_url",
        str(model_path),

        "--output_url",
        str(save_path),

        "--predict_mask_num",
        str(predict_mask_num),

        "--mask_prob",
        str(mask_prob),
    ]

    print("\nCommand:")
    print(
        subprocess.list2cmdline(command)
    )
    print()

    try:
        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "LYS-MUT prediction failed."
        ) from error


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    input_file = (
        args.input
        .expanduser()
        .resolve()
    )

    model_dir = (
        args.model_dir
        .expanduser()
        .resolve()
    )

    mpbert_dir = (
        args.mpbert_dir
        .expanduser()
        .resolve()
    )

    vocab_file = (
        args.vocab_file
        .expanduser()
        .resolve()
    )

    # Target-specific model generated during LYS-MUT training.
    model_path = (
        model_dir /
        "mask_Best_Model.ckpt"
    )

    # MP-BERT files.
    mpbert_script = (
        mpbert_dir /
        "mpbert_mask.py"
    )

    config_file = (
        mpbert_dir /
        "config_1024.yaml"
    )

    # Prediction output directory.
    save_path = (
        model_dir /
        "prediction"
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    check_required_files(
        input_file=input_file,
        model_path=model_path,
        mpbert_script=mpbert_script,
        config_file=config_file,
        vocab_file=vocab_file,
    )

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LYS-MUT prediction")
    print("=" * 70)

    print(
        f"Input FASTA      : {input_file}"
    )

    print(
        f"Model            : {model_path}"
    )

    print(
        f"MP-BERT directory: {mpbert_dir}"
    )

    print(
        f"Vocabulary file  : {vocab_file}"
    )

    print(
        f"Output directory : {save_path}"
    )

    print(
        f"Device ID        : {args.device}"
    )

    print(
        f"Generation number: {args.predict_mask_num}"
    )

    print(
        f"Mask probability : {args.mask_prob}"
    )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    run_prediction(
        input_file=input_file,
        model_path=model_path,
        save_path=save_path,
        mpbert_script=mpbert_script,
        config_file=config_file,
        vocab_file=vocab_file,
        device=args.device,
        predict_mask_num=args.predict_mask_num,
        mask_prob=args.mask_prob,
    )

    print("\n" + "=" * 70)
    print("Prediction completed successfully.")
    print("=" * 70)

    print(
        f"\nResults saved to: {save_path}"
    )


if __name__ == "__main__":
    main()
