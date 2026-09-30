#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare MindRecord datasets and train the LYS-MUT target-specific model.

Workflow
--------
1. Convert train.fasta, val.fasta, and test.fasta into the dataset
   format required by MP-BERT masked-model training.
2. Fine-tune the MP-BERT masked model using the target-specific
   homologous sequence dataset.

Example
-------
python step_3_run_train.py \
    --data_path ./example/example_seq/train_data \
    --mpbert_dir /path/to/MP-BERT-v3 \
    --vocab_file /path/to/vocab_v2.txt \
    --checkpoint /path/to/pretrained_model.ckpt \
    --device 0

Training parameters used in this study
--------------------------------------
Maximum sequence length : 1024
Epochs                  : 200
Early stopping rounds   : 50
Training batch size     : 32
Frozen BERT             : False
Task                    : mask
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
            "Prepare the target-specific dataset and train "
            "the LYS-MUT masked sequence model."
        )
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help=(
            "Path to the training-data directory containing "
            "train.fasta, val.fasta, and test.fasta."
        ),
    )

    parser.add_argument(
        "--mpbert_dir",
        type=Path,
        required=True,
        help=(
            "Path to the MP-BERT directory containing "
            "mpbert_mask.py, config_1024.yaml, and "
            "generate_dataset/generate_seq_for_mask.py."
        ),
    )

    parser.add_argument(
        "--vocab_file",
        type=Path,
        required=True,
        help="Path to vocab_v2.txt.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the pretrained MP-BERT initialization "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Ascend device ID used for training (default: 0).",
    )

    return parser.parse_args()


# ============================================================
# 2. Check required files
# ============================================================

def check_required_files(
    data_path,
    mpbert_dir,
    vocab_file,
    checkpoint,
):
    """Check all required input files before running the pipeline."""

    required_datasets = [
        data_path / "train.fasta",
        data_path / "val.fasta",
        data_path / "test.fasta",
    ]

    generate_script = (
        mpbert_dir
        / "generate_dataset"
        / "generate_seq_for_mask.py"
    )

    training_script = (
        mpbert_dir
        / "mpbert_mask.py"
    )

    config_file = (
        mpbert_dir
        / "config_1024.yaml"
    )

    required_files = (
        required_datasets
        + [
            generate_script,
            training_script,
            config_file,
            vocab_file,
            checkpoint,
        ]
    )

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

    return (
        generate_script,
        training_script,
        config_file,
    )


# ============================================================
# 3. Generate training datasets
# ============================================================

def generate_dataset(
    data_path,
    generate_script,
    vocab_file,
):
    """Convert FASTA datasets into the format required by MP-BERT."""

    print("\n" + "=" * 70)
    print("STEP 1: Generate MP-BERT datasets")
    print("=" * 70)

    log_file = (
        data_path
        / "data_process_log.log"
    )

    error_file = (
        data_path
        / "data_process_sys.log"
    )

    command = [
        sys.executable,
        str(generate_script),

        "--data_dir",
        str(data_path),

        "--vocab_file",
        str(vocab_file),

        "--output_dir",
        str(data_path),

        "--max_seq_length",
        "1024",

        "--do_train",
        "True",

        "--do_eval",
        "True",

        "--do_test",
        "True",
    ]

    print("\nRunning command:")
    print(
        subprocess.list2cmdline(command)
    )

    print(
        f"\nStandard output: {log_file}"
    )

    print(
        f"Error log      : {error_file}"
    )

    with (
        open(log_file, "w") as stdout_file,
        open(error_file, "w") as stderr_file,
    ):
        try:
            subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                check=True,
            )

        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Dataset generation failed. "
                f"Please check:\n"
                f"  {log_file}\n"
                f"  {error_file}"
            ) from error

    print(
        "\nDataset generation completed successfully."
    )


# ============================================================
# 4. Train target-specific model
# ============================================================

def train_model(
    data_path,
    training_script,
    config_file,
    checkpoint,
    device,
):
    """Start target-specific MP-BERT masked-model training."""

    print("\n" + "=" * 70)
    print("STEP 2: Train target-specific LYS-MUT model")
    print("=" * 70)

    log_file = (
        data_path
        / "train_log.log"
    )

    error_file = (
        data_path
        / "train_sys.log"
    )

    command = [
        sys.executable,
        str(training_script),

        "--config_path",
        str(config_file),

        "--do_train",
        "True",

        "--do_eval",
        "True",

        "--description",
        "sequence",

        "--epoch_num",
        "200",

        "--early_stopping_rounds",
        "50",

        "--frozen_bert",
        "False",

        "--device_id",
        str(device),

        "--data_url",
        str(data_path),

        "--load_checkpoint_url",
        str(checkpoint),

        "--output_url",
        str(data_path),

        "--task_name",
        "mask",

        "--train_batch_size",
        "32",
    ]

    print("\nTraining command:")
    print(
        subprocess.list2cmdline(command)
    )

    print(
        f"\nTraining log: {log_file}"
    )

    print(
        f"Error log   : {error_file}"
    )

    # Equivalent to the original:
    #
    # nohup python ... > train_log.log 2> train_sys.log &
    #
    # The training process continues independently after this
    # wrapper script exits.

    stdout_file = open(
        log_file,
        "w"
    )

    stderr_file = open(
        error_file,
        "w"
    )

    try:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise

    stdout_file.close()
    stderr_file.close()

    print(
        "\nModel training has been started in the background."
    )

    print(
        f"Process ID: {process.pid}"
    )

    print(
        "\nMonitor training with:"
    )

    print(
        f"tail -f {log_file}"
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    data_path = (
        args.data_path
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

    checkpoint = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    (
        generate_script,
        training_script,
        config_file,
    ) = check_required_files(
        data_path=data_path,
        mpbert_dir=mpbert_dir,
        vocab_file=vocab_file,
        checkpoint=checkpoint,
    )

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LYS-MUT training pipeline")
    print("=" * 70)

    print(
        f"Data path             : {data_path}"
    )

    print(
        f"MP-BERT directory     : {mpbert_dir}"
    )

    print(
        f"Vocabulary file       : {vocab_file}"
    )

    print(
        f"Initialization model  : {checkpoint}"
    )

    print(
        f"Device ID             : {args.device}"
    )

    # ========================================================
    # STEP 1
    # Generate datasets
    # ========================================================

    generate_dataset(
        data_path=data_path,
        generate_script=generate_script,
        vocab_file=vocab_file,
    )

    # ========================================================
    # STEP 2
    # Train model
    # ========================================================

    train_model(
        data_path=data_path,
        training_script=training_script,
        config_file=config_file,
        checkpoint=checkpoint,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("LYS-MUT training pipeline started successfully.")
    print("=" * 70)

    print(
        "\nDataset preparation has completed and model training "
        "is running in the background."
    )


if __name__ == "__main__":
    main()
