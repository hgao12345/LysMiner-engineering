#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate training datasets for MPB-CLS.

Workflow
--------
1. Use jackhmmer to search UniRef90 using one input protein sequence.
2. Extract matched protein sequences from UniRef90.
3. Save the matched sequences as a FASTA file.
4. Split the sequences into training, validation, and test datasets.

Dataset split
-------------
Train : 64%
Val   : 16%
Test  : 20%

Output structure
----------------
output_dir/
├── input_name.hmmer.out.o
├── input_name.hmmer.tblout
├── input_name.hmmer.domtblout
├── input_name.hmmer.fasta
│
└── input_name/
    └── train_data/
        ├── train.fasta
        ├── val.fasta
        └── test.fasta

Example
-------
python generate_dataset.py \
    --fasta_file example/LYS.fasta \
    --uniref /path/to/uniref90.fasta \
    --output results \
    --cpu 8
"""

import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# 1. Parse arguments
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description="Generate training datasets for MPB-CLS."
    )

    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "HMMER score threshold coefficient relative to the "
            "query sequence length (default: 0.5)."
        ),
    )

    parser.add_argument(
        "-F",
        "--fasta_file",
        type=Path,
        required=True,
        help=(
            "Input protein FASTA file containing exactly one "
            "protein sequence."
        ),
    )

    parser.add_argument(
        "-U",
        "--uniref",
        type=Path,
        required=True,
        help="Path to the UniRef90 FASTA database.",
    )

    parser.add_argument(
        "-O",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. If not specified, results are saved "
            "in the directory containing the input FASTA file."
        ),
    )

    parser.add_argument(
        "-A",
        "--aggregation",
        type=int,
        default=1,
        help=(
            "Number of times the training dataset is duplicated "
            "for data aggregation (default: 1)."
        ),
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=8,
        help=(
            "Number of CPU threads used by jackhmmer "
            "(default: 8)."
        ),
    )

    return parser.parse_args()


# ============================================================
# 2. Check input files and dependencies
# ============================================================

def check_requirements(args):
    """Check input files, arguments, and external dependencies."""

    if shutil.which("jackhmmer") is None:
        raise FileNotFoundError(
            "jackhmmer was not found. Please install HMMER and "
            "ensure that 'jackhmmer' is available in your system PATH."
        )

    if not args.fasta_file.exists():
        raise FileNotFoundError(
            f"Input FASTA file not found: {args.fasta_file}"
        )

    if not args.uniref.exists():
        raise FileNotFoundError(
            f"UniRef90 FASTA file not found: {args.uniref}"
        )

    if not 0 < args.threshold <= 1:
        raise ValueError(
            "--threshold must be greater than 0 and less than or equal to 1."
        )

    if args.aggregation < 1:
        raise ValueError(
            "--aggregation must be an integer greater than or equal to 1."
        )

    if args.cpu < 1:
        raise ValueError(
            "--cpu must be an integer greater than or equal to 1."
        )


# ============================================================
# 3. Read FASTA
# ============================================================

def read_fasta(fasta_file):
    """Read a FASTA file and return [[id, sequence], ...]."""

    sequences = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                sequences.append(
                    [
                        line[1:],
                        "",
                    ]
                )

            else:
                if not sequences:
                    raise ValueError(
                        f"Invalid FASTA format in {fasta_file}: "
                        "sequence found before FASTA header."
                    )

                sequences[-1][1] += line

    if len(sequences) == 0:
        raise ValueError(
            f"No protein sequences were found in: {fasta_file}"
        )

    return sequences


# ============================================================
# 4. Run jackhmmer
# ============================================================

def run_hmmer(
    input_fasta,
    input_name,
    output_dir,
    args,
):
    """Run jackhmmer against the UniRef90 database."""

    protein = read_fasta(
        input_fasta
    )

    # Input FASTA must contain exactly one query protein.
    if len(protein) != 1:
        raise ValueError(
            "Input FASTA must contain exactly one protein sequence."
        )

    seq_len = len(
        protein[0][1]
    )

    seq_threshold = int(
        seq_len * args.threshold
    )

    print("\n" + "=" * 70)
    print("STEP 1: Run jackhmmer")
    print("=" * 70)

    print(f"Input FASTA       : {input_fasta}")
    print(f"Sequence length   : {seq_len}")
    print(f"Threshold         : {args.threshold}")
    print(f"Sequence threshold: {seq_threshold}")
    print(f"UniRef90          : {args.uniref}")
    print(f"CPU threads       : {args.cpu}")

    hmmer_output = (
        output_dir /
        f"{input_name}.hmmer.out.o"
    )

    tblout_file = (
        output_dir /
        f"{input_name}.hmmer.tblout"
    )

    domtblout_file = (
        output_dir /
        f"{input_name}.hmmer.domtblout"
    )

    command = [
        "jackhmmer",

        "-N",
        "5",

        "-o",
        str(hmmer_output),

        "--tblout",
        str(tblout_file),

        "--domtblout",
        str(domtblout_file),

        "--notextw",

        "-T",
        str(seq_threshold),

        "--domT",
        str(seq_threshold),

        "--incT",
        str(seq_threshold),

        "--incdomT",
        str(seq_threshold),

        "--cpu",
        str(args.cpu),

        str(input_fasta),

        str(args.uniref),
    ]

    print("\nRunning command:")
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
            "jackhmmer failed. Please check the HMMER installation, "
            "input FASTA file, UniRef90 database, and command parameters."
        ) from error

    if not tblout_file.exists():
        raise FileNotFoundError(
            f"HMMER tblout file was not generated: {tblout_file}"
        )

    if not domtblout_file.exists():
        raise FileNotFoundError(
            f"HMMER domtblout file was not generated: {domtblout_file}"
        )

    print("\njackhmmer search completed.")


# ============================================================
# 5. Read HMMER tblout
# ============================================================

def read_tblout(tblout_file):
    """Read sequence IDs from the HMMER tblout file."""

    print("\n" + "=" * 70)
    print("STEP 2: Read HMMER results")
    print("=" * 70)

    print(
        f"Reading tblout file: {tblout_file}",
        flush=True,
    )

    uniref90_ids = []

    with open(tblout_file, "r") as f:
        for line in tqdm(f):

            line = line.strip()

            # Skip comments and empty lines.
            if not line:
                continue

            if line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) == 0:
                continue

            seq_id = fields[0]

            uniref90_ids.append(
                seq_id
            )

    # Remove duplicate IDs while preserving order.
    uniref90_ids = list(
        dict.fromkeys(uniref90_ids)
    )

    if len(uniref90_ids) == 0:
        raise ValueError(
            "No UniRef90 sequences were found by jackhmmer."
        )

    print(
        f"Found UniRef90 IDs: {len(uniref90_ids)}",
        flush=True,
    )

    return uniref90_ids


# ============================================================
# 6. Extract matched sequences from UniRef90
# ============================================================

def extract_tblout(
    input_name,
    output_dir,
    uniref_file,
):
    """Extract jackhmmer hits from the UniRef90 FASTA database."""

    tblout_file = (
        output_dir /
        f"{input_name}.hmmer.tblout"
    )

    uniref_ids = read_tblout(
        tblout_file
    )

    # Convert IDs to a set for faster lookup.
    uniref_ids = set(
        uniref_ids
    )

    print("\nExtracting sequences from UniRef90...")

    seq_result = []

    temp_name = None
    temp_seq = ""

    with open(uniref_file, "r") as f:
        for line in tqdm(f):

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                # Save the previous sequence if it is a HMMER hit.
                if (
                    temp_name is not None
                    and temp_name in uniref_ids
                ):
                    seq_result.append(
                        [
                            temp_name,
                            temp_seq,
                        ]
                    )

                # Start reading a new sequence.
                temp_name = (
                    line[1:]
                    .split()[0]
                )

                temp_seq = ""

            else:
                temp_seq += line

    # Save the final sequence.
    if (
        temp_name is not None
        and temp_name in uniref_ids
    ):
        seq_result.append(
            [
                temp_name,
                temp_seq,
            ]
        )

    print(
        f"Sequences extracted from UniRef90: {len(seq_result)}",
        flush=True,
    )

    if len(seq_result) == 0:
        raise ValueError(
            "No sequences could be extracted from UniRef90."
        )

    # Check whether some HMMER hits were not found in the FASTA file.
    extracted_ids = {
        seq_id
        for seq_id, _ in seq_result
    }

    missing_ids = (
        uniref_ids - extracted_ids
    )

    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} HMMER hit IDs "
            "were not found in the UniRef90 FASTA file."
        )

    return seq_result


# ============================================================
# 7. Write HMMER result FASTA
# ============================================================

def write_fasta(
    seq_list,
    output_dir,
    input_name,
):
    """Save extracted HMMER hit sequences to FASTA."""

    print("\n" + "=" * 70)
    print("STEP 3: Save HMMER sequences")
    print("=" * 70)

    output_file = (
        output_dir /
        f"{input_name}.hmmer.fasta"
    )

    with open(output_file, "w") as f:
        for seq_id, sequence in seq_list:

            f.write(
                f">{seq_id}\n"
            )

            f.write(
                f"{sequence}\n"
            )

    print(
        f"HMMER FASTA saved to: {output_file}"
    )

    return output_file


# ============================================================
# 8. Save FASTA dataset
# ============================================================

def save_dataset(data, output_file):
    """Save a dataset to FASTA format."""

    with open(output_file, "w") as f:
        for seq_id, sequence in data:

            f.write(
                f">{seq_id}\n"
            )

            f.write(
                f"{sequence}\n"
            )


# ============================================================
# 9. Generate train / validation / test datasets
# ============================================================

def make_dataset(
    output_dir,
    input_name,
    aggregation,
):
    """Split HMMER sequences into train, validation, and test sets."""

    print("\n" + "=" * 70)
    print("STEP 4: Generate train / validation / test datasets")
    print("=" * 70)

    hmmer_results = (
        output_dir /
        f"{input_name}.hmmer.fasta"
    )

    # --------------------------------------------------------
    # Read HMMER sequences
    # --------------------------------------------------------

    hmm_data = read_fasta(
        hmmer_results
    )

    print(
        f"Total sequences before duplicate check: {len(hmm_data)}"
    )

    # --------------------------------------------------------
    # Check duplicate sequences
    # --------------------------------------------------------

    sequences = [
        sequence
        for _, sequence in hmm_data
    ]

    if len(hmm_data) != len(set(sequences)):
        raise ValueError(
            "Duplicate protein sequences were found in the "
            "HMMER results."
        )

    # A minimum number of sequences is needed for the two-step split.
    if len(hmm_data) < 5:
        raise ValueError(
            "Too few sequences were retrieved to generate "
            "train, validation, and test datasets."
        )

    hmm_data = np.array(
        hmm_data,
        dtype=object,
    )

    print(
        f"Total sequences used for dataset: {len(hmm_data)}"
    )

    # ========================================================
    # First split
    #
    # 80% -> train + validation
    # 20% -> test
    # ========================================================

    train_val, test = train_test_split(
        hmm_data,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # ========================================================
    # Second split
    #
    # train_val = 80% of total
    #
    # 80% of train_val -> train
    # 20% of train_val -> validation
    #
    # Final ratio:
    #
    # train = 64%
    # val   = 16%
    # test  = 20%
    # ========================================================

    train, val = train_test_split(
        train_val,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # Keep original dataset sizes before optional aggregation.
    original_train_size = len(train)
    original_val_size = len(val)
    original_test_size = len(test)

    # ========================================================
    # Training data aggregation
    # ========================================================

    if aggregation > 1:
        train = np.concatenate(
            [
                train
                for _ in range(aggregation)
            ],
            axis=0,
        )

    # Reproducible shuffle of the final training dataset.
    rng = np.random.default_rng(42)

    rng.shuffle(
        train,
        axis=0,
    )

    # ========================================================
    # Output directory
    # ========================================================

    save_path = (
        output_dir /
        input_name /
        "train_data"
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Save datasets
    # ========================================================

    train_file = (
        save_path /
        "train.fasta"
    )

    val_file = (
        save_path /
        "val.fasta"
    )

    test_file = (
        save_path /
        "test.fasta"
    )

    save_dataset(
        train,
        train_file,
    )

    save_dataset(
        val,
        val_file,
    )

    save_dataset(
        test,
        test_file,
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\nDataset split completed.")
    print("-" * 70)

    original_total = (
        original_train_size
        + original_val_size
        + original_test_size
    )

    print(
        f"Original train sequences      : {original_train_size}"
    )

    print(
        f"Validation sequences          : {original_val_size}"
    )

    print(
        f"Test sequences                : {original_test_size}"
    )

    print(
        f"Original train ratio          : "
        f"{original_train_size / original_total * 100:.2f}%"
    )

    print(
        f"Validation ratio              : "
        f"{original_val_size / original_total * 100:.2f}%"
    )

    print(
        f"Test ratio                    : "
        f"{original_test_size / original_total * 100:.2f}%"
    )

    if aggregation > 1:
        print(
            f"Training aggregation          : {aggregation}x"
        )

        print(
            f"Final training entries        : {len(train)}"
        )

    print("\nFiles saved to:")

    print(
        f"Train: {train_file}"
    )

    print(
        f"Val  : {val_file}"
    )

    print(
        f"Test : {test_file}"
    )


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    args = parse_args()

    # Resolve user-provided paths.
    input_fasta = (
        args.fasta_file
        .expanduser()
        .resolve()
    )

    uniref_file = (
        args.uniref
        .expanduser()
        .resolve()
    )

    # Update paths stored in args for validation.
    args.fasta_file = input_fasta
    args.uniref = uniref_file

    # --------------------------------------------------------
    # Check files and dependencies
    # --------------------------------------------------------

    check_requirements(
        args
    )

    # --------------------------------------------------------
    # Get input filename without extension
    #
    # Example:
    #
    # /data/example/LYS.fasta
    #
    # becomes:
    #
    # LYS
    # --------------------------------------------------------

    input_name = (
        input_fasta.stem
    )

    # --------------------------------------------------------
    # Determine output directory
    # --------------------------------------------------------

    if args.output is None:
        output_dir = (
            input_fasta.parent
        )

    else:
        output_dir = (
            args.output
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Pipeline information
    # ========================================================

    print("\n" + "=" * 70)
    print("MPB-CLS dataset generation pipeline")
    print("=" * 70)

    print(
        f"Input FASTA     : {input_fasta}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"UniRef90        : {uniref_file}"
    )

    print(
        f"HMMER threshold : {args.threshold}"
    )

    print(
        f"CPU threads     : {args.cpu}"
    )

    print(
        f"Data aggregation: {args.aggregation}"
    )

    # ========================================================
    # STEP 1
    # Run jackhmmer
    # ========================================================

    run_hmmer(
        input_fasta=input_fasta,
        input_name=input_name,
        output_dir=output_dir,
        args=args,
    )

    # ========================================================
    # STEP 2
    # Extract matched UniRef90 sequences
    # ========================================================

    uniref90_result = extract_tblout(
        input_name=input_name,
        output_dir=output_dir,
        uniref_file=uniref_file,
    )

    # ========================================================
    # STEP 3
    # Save matched sequences
    # ========================================================

    write_fasta(
        seq_list=uniref90_result,
        output_dir=output_dir,
        input_name=input_name,
    )

    # ========================================================
    # STEP 4
    # Generate one train / validation / test dataset
    # ========================================================

    make_dataset(
        output_dir=output_dir,
        input_name=input_name,
        aggregation=args.aggregation,
    )

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
