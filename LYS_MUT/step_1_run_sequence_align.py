#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate training dataset for MPB-CLS.

Workflow:
1. Use jackhmmer to search UniRef90 using one input protein sequence.
2. Extract matched sequences from UniRef90.
3. Save the matched sequences as a FASTA file.
4. Split the sequences into train / validation / test datasets.

Dataset split:
    Train : 64%
    Val   : 16%
    Test  : 20%

Output structure:
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
"""


from argparse import ArgumentParser
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import numpy as np


# ============================================================
# 1. Parse arguments
# ============================================================

def parse_args():

    parser = ArgumentParser(
        description="Generate training dataset for MPB-CLS"
    )

    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=0.5,
        help="Set HMMER threshold from 0-1 (0-100%%)"
    )

    parser.add_argument(
        "-F",
        "--fasta_file",
        type=str,
        required=True,
        help="Input protein FASTA file containing only one sequence"
    )

    parser.add_argument(
        "-U",
        "--uniref",
        type=str,
        default="/data1/pub_data/UniProt/uniref90.fasta",
        help="UniRef90 FASTA file"
    )

    parser.add_argument(
        "-O",
        "--output",
        type=str,
        required=False,
        default=None,
        help="Output directory"
    )

    parser.add_argument(
        "-A",
        "--aggregation",
        type=int,
        required=False,
        default=1,
        help="Training data aggregation times"
    )

    args_opt = parser.parse_args()

    return args_opt


# ============================================================
# 2. Read FASTA
# ============================================================

def read_fasta(fasta_file):

    seq = []

    with open(fasta_file) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                seq.append([
                    line[1:],
                    ""
                ])

            else:

                seq[-1][-1] += line

    return seq


# ============================================================
# 3. Run jackhmmer
# ============================================================

def run_hmmer(
    input_fasta,
    input_name,
    output_dir,
    args
):

    protein = read_fasta(
        input_fasta
    )

    # Input FASTA should contain only one protein.
    assert len(protein) == 1, \
        "Input FASTA must contain exactly one protein sequence."

    seq_len = len(
        protein[0][1]
    )

    seq_threshold = int(
        seq_len * args.threshold
    )

    print("\n" + "=" * 70)
    print("STEP 1: Run jackhmmer")
    print("=" * 70)

    print(
        "Input FASTA:",
        input_fasta
    )

    print(
        "Sequence length:",
        seq_len
    )

    print(
        "Sequence threshold:",
        seq_threshold
    )

    print(
        "UniRef90:",
        args.uniref
    )

    hmmer_script = [

        "jackhmmer",

        "-N",
        "5",

        "-o",
        os.path.join(
            output_dir,
            input_name + ".hmmer.out.o"
        ),

        "--tblout",
        os.path.join(
            output_dir,
            input_name + ".hmmer.tblout"
        ),

        "--domtblout",
        os.path.join(
            output_dir,
            input_name + ".hmmer.domtblout"
        ),

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
        "50",

        input_fasta,

        args.uniref
    ]

    hmmer_script = " ".join(
        hmmer_script
    )

    print("\nRunning command:")
    print(hmmer_script)
    print()

    return_code = os.system(
        hmmer_script
    )

    if return_code != 0:

        raise RuntimeError(
            "jackhmmer failed."
        )

    domtblout_file = os.path.join(
        output_dir,
        input_name + ".hmmer.domtblout"
    )

    assert os.path.exists(
        domtblout_file
    ), f"HMMER output not found: {domtblout_file}"


# ============================================================
# 4. Read HMMER tblout
# ============================================================

def read_tblout(tblout_file):

    print("\n" + "=" * 70)
    print("STEP 2: Read HMMER results")
    print("=" * 70)

    print(
        "Reading tblout file:",
        tblout_file,
        flush=True
    )

    uniref90_id = []

    with open(tblout_file) as f:

        for line in tqdm(f):

            line = line.strip()

            # Skip comments and empty lines.
            if not line:
                continue

            if line.startswith("#"):
                continue

            line = line.split()

            seq_id = line[0]

            uniref90_id.append(
                seq_id
            )

    # Remove duplicate IDs.
    uniref90_id = list(
        set(uniref90_id)
    )

    if len(uniref90_id) == 0:

        raise ValueError(
            "No UniRef90 sequences were found by HMMER."
        )

    print(
        "Found UniRef90 IDs:",
        len(uniref90_id),
        flush=True
    )

    return uniref90_id


# ============================================================
# 5. Extract matched sequences from UniRef90
# ============================================================

def extract_tblout(
    input_name,
    output_dir,
    args
):

    tblout_file = os.path.join(
        output_dir,
        input_name + ".hmmer.tblout"
    )

    uniref_id = read_tblout(
        tblout_file
    )

    # Convert to set for faster lookup.
    uniref_id = set(
        uniref_id
    )

    print("\nExtracting sequences from UniRef90...")

    seq_result = []

    temp_name = None
    temp_seq = ""

    with open(args.uniref, "r") as f:

        for line in tqdm(f):

            line = line.strip()

            if line.startswith(">"):

                # Save previous sequence if it is a HMMER hit.
                if (
                    temp_name is not None
                    and temp_name in uniref_id
                ):

                    seq_result.append([
                        ">" + temp_name,
                        temp_seq
                    ])

                # Start reading a new sequence.
                temp_name = (
                    line[1:]
                    .split()[0]
                )

                temp_seq = ""

            else:

                temp_seq += line

    # Save final sequence.
    if (
        temp_name is not None
        and temp_name in uniref_id
    ):

        seq_result.append([
            ">" + temp_name,
            temp_seq
        ])

    print(
        "Sequences extracted from UniRef90:",
        len(seq_result),
        flush=True
    )

    if len(seq_result) == 0:

        raise ValueError(
            "No sequences could be extracted from UniRef90."
        )

    return seq_result


# ============================================================
# 6. Write HMMER result FASTA
# ============================================================

def write_fasta(
    seq_list,
    output_dir,
    input_name
):

    print("\n" + "=" * 70)
    print("STEP 3: Save HMMER sequences")
    print("=" * 70)

    output_file = os.path.join(
        output_dir,
        input_name + ".hmmer.fasta"
    )

    with open(
        output_file,
        "w"
    ) as f:

        for seq in seq_list:

            f.write(
                seq[0] + "\n"
            )

            f.write(
                seq[1] + "\n"
            )

    print(
        "HMMER FASTA saved to:",
        output_file
    )


# ============================================================
# 7. Generate train / validation / test dataset
# ============================================================

def make_dataset(
    output_dir,
    input_name,
    args
):

    print("\n" + "=" * 70)
    print("STEP 4: Generate train / validation / test dataset")
    print("=" * 70)

    data_aggregation_time = (
        args.aggregation
    )

    hmmer_results = os.path.join(
        output_dir,
        input_name + ".hmmer.fasta"
    )

    # --------------------------------------------------------
    # Read HMMER sequences
    # --------------------------------------------------------

    hmm_data = read_fasta(
        hmmer_results
    )

    print(
        "Total sequences before duplicate check:",
        len(hmm_data)
    )

    # --------------------------------------------------------
    # Check duplicate sequences
    # --------------------------------------------------------

    sequences = [
        i[1]
        for i in hmm_data
    ]

    assert len(hmm_data) == len(set(sequences)), (
        "Duplicate protein sequences were found "
        "in the HMMER results."
    )

    hmm_data = np.array(
        hmm_data,
        dtype=object
    )

    print(
        "Total sequences used for dataset:",
        len(hmm_data)
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

        shuffle=True
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

        shuffle=True
    )

    # ========================================================
    # Training data aggregation
    # ========================================================

    if data_aggregation_time > 1:

        train = np.concatenate(

            [
                train
                for _ in range(
                    data_aggregation_time
                )
            ],

            axis=0
        )

    # Shuffle training data.
    np.random.shuffle(
        train
    )

    # ========================================================
    # Output directory
    # ========================================================

    save_path = os.path.join(
        output_dir,
        input_name,
        "train_data"
    )

    os.makedirs(
        save_path,
        exist_ok=True
    )

    # ========================================================
    # Save train.fasta
    # ========================================================

    train_file = os.path.join(
        save_path,
        "train.fasta"
    )

    with open(
        train_file,
        "w"
    ) as f:

        for seq_id, sequence in train:

            f.write(
                ">" +
                str(seq_id) +
                "\n"
            )

            f.write(
                str(sequence) +
                "\n"
            )

    # ========================================================
    # Save val.fasta
    # ========================================================

    val_file = os.path.join(
        save_path,
        "val.fasta"
    )

    with open(
        val_file,
        "w"
    ) as f:

        for seq_id, sequence in val:

            f.write(
                ">" +
                str(seq_id) +
                "\n"
            )

            f.write(
                str(sequence) +
                "\n"
            )

    # ========================================================
    # Save test.fasta
    # ========================================================

    test_file = os.path.join(
        save_path,
        "test.fasta"
    )

    with open(
        test_file,
        "w"
    ) as f:

        for seq_id, sequence in test:

            f.write(
                ">" +
                str(seq_id) +
                "\n"
            )

            f.write(
                str(sequence) +
                "\n"
            )

    # ========================================================
    # Summary
    # ========================================================

    print("\nDataset split completed.")
    print("-" * 70)

    print(
        f"Train sequences      : {len(train)}"
    )

    print(
        f"Validation sequences : {len(val)}"
    )

    print(
        f"Test sequences       : {len(test)}"
    )

    total = (
        len(train)
        + len(val)
        + len(test)
    )

    # Only report percentages directly when aggregation = 1,
    # because aggregation duplicates training sequences.
    if data_aggregation_time == 1:

        print(
            f"Train ratio          : "
            f"{len(train) / total * 100:.2f}%"
        )

        print(
            f"Validation ratio     : "
            f"{len(val) / total * 100:.2f}%"
        )

        print(
            f"Test ratio           : "
            f"{len(test) / total * 100:.2f}%"
        )

    print("\nFiles saved to:")

    print(
        "Train:",
        train_file
    )

    print(
        "Val  :",
        val_file
    )

    print(
        "Test :",
        test_file
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    args = parse_args()

    input_dir = args.fasta_file

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

    input_file_name = os.path.splitext(
        os.path.basename(
            input_dir
        )
    )[0]

    # --------------------------------------------------------
    # Determine output directory
    # --------------------------------------------------------

    if args.output is None:

        output_dir = os.path.dirname(
            os.path.abspath(
                input_dir
            )
        )

    else:

        output_dir = os.path.abspath(
            args.output
        )

    # Create output directory if necessary.
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("Dataset generation pipeline")
    print("=" * 70)

    print(
        "Input FASTA:",
        input_dir
    )

    print(
        "Output directory:",
        output_dir
    )

    print(
        "UniRef90:",
        args.uniref
    )

    print(
        "HMMER threshold:",
        args.threshold
    )

    print(
        "Data aggregation:",
        args.aggregation
    )

    # ========================================================
    # STEP 1
    # Run jackhmmer
    # ========================================================

    run_hmmer(
        input_dir,
        input_file_name,
        output_dir,
        args
    )

    # ========================================================
    # STEP 2
    # Extract matched UniRef90 sequences
    # ========================================================

    uniref90_result = extract_tblout(
        input_file_name,
        output_dir,
        args
    )

    # ========================================================
    # STEP 3
    # Save matched sequences
    # ========================================================

    write_fasta(
        uniref90_result,
        output_dir,
        input_file_name
    )

    # ========================================================
    # STEP 4
    # Generate ONE train / val / test dataset
    # ========================================================

    make_dataset(
        output_dir,
        input_file_name,
        args
    )

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)