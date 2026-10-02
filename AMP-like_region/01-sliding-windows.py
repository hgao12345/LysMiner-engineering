import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--work_path',
    type=str,
    required=True,
    help='Working directory containing the input CSV file'
)

parser.add_argument(
    '--seq_name',
    type=str,
    required=True,
    help='Name of the input sequence file without the .csv extension'
)

args = parser.parse_args()

work_path = args.work_path
seq_name = args.seq_name


os.chdir(work_path)


def process_sequences(
    sequences,
    ids_,
    window_size=13,
    step_size=1,
    output_file=None
):
    if output_file is None:
        output_file = f"{seq_name}_sliding_window_sequences13AA.csv"

    # Initialize lists to hold the IDs and sequences
    new_ids = []
    seqs = []

    # Loop through each sequence
    for seq_index, sequence in enumerate(sequences):
        # Loop through the sequence with the sliding window
        for i in range(0, len(sequence) - window_size + 1, step_size):
            new_ids.append(f"{ids_[seq_index]}_{i+1}")
            seqs.append(sequence[i:i+window_size])

    # Create a DataFrame
    df = pd.DataFrame({'id': new_ids, 'seq': seqs})

    # Save to a CSV file
    df.to_csv(output_file, index=False)


# Read input CSV
file_ = pd.read_csv(f"{seq_name}.csv", header=0)

ids_ = list(file_.iloc[:, 0])
sequences = list(file_.iloc[:, 1])

process_sequences(sequences, ids_)
