import pandas as pd
from pathlib import Path
import argparse


# ==========================
# Arguments
# ==========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--workdir",
    type=str,
    required=True,
    help="Working directory containing the prediction results"
)

parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Input prediction CSV file"
)

parser.add_argument(
    "--window_size",
    type=int,
    default=13,
    help="Sliding window size (default: 13)"
)

args = parser.parse_args()


workdir = Path(args.workdir)
input_file = args.input_file
window_size = args.window_size


# ==========================
# Generate output filenames
# ==========================

input_stem = Path(input_file).stem

suffix = "_sliding_window_sequences13AA_predict_result"

if input_stem.endswith(suffix):
    output_prefix = input_stem[:-len(suffix)]
else:
    output_prefix = input_stem

coverage_output = f"{output_prefix}_protein_coverage_ratios.csv"
windows_output = f"{output_prefix}_all_windows.csv"

# ==========================
# Read prediction results
# ==========================

file = workdir / input_file

df = pd.read_csv(file)
df = df[["id", "seq", "pred_label"]].copy()

# pred_label == 1 is AMP-positive
df["positive"] = df["pred_label"] == 1

print("Total windows:", len(df))
print("Positive windows:", df["positive"].sum())


# ==========================
# Extract protein name and
# window start position from ID
#
# RL6_FRATF_1
# -> protein_name = RL6_FRATF
# -> start_pos = 1
# ==========================

df[["protein_name", "start_pos"]] = (
    df["id"].str.rsplit("_", n=1, expand=True)
)

df["start_pos"] = df["start_pos"].astype(int)


# ==========================
# Calculate positive-region
# coverage for each protein
# ==========================

results = []

for protein_name, group in df.groupby("protein_name"):

    # Assume window positions start from 1 and the final
    # complete window reaches the end of the protein
    protein_length = int(
        group["start_pos"].max() + window_size - 1
    )

    # Select windows predicted as AMP-positive
    positive_group = group[group["positive"]]

    # Calculate the union of residues covered by AMP-positive windows
    all_positions = set()

    for start_pos in positive_group["start_pos"]:
        all_positions.update(
            range(start_pos, start_pos + window_size)
        )

    unique_positions = len(all_positions)
    coverage_ratio = unique_positions / protein_length

    results.append(
        {
            "protein_name": protein_name,
            "protein_length": protein_length,
            "positive_fragments": len(positive_group),
            "unique_positive_positions": unique_positions,
            "coverage_ratio": coverage_ratio,
        }
    )


# ==========================
# Save protein-level results
# ==========================

result_df = pd.DataFrame(
    results,
    columns=[
        "protein_name",
        "protein_length",
        "positive_fragments",
        "unique_positive_positions",
        "coverage_ratio",
    ],
)

result_df = result_df.sort_values(
    "coverage_ratio",
    ascending=False,
)

result_df.to_csv(
    workdir / coverage_output,
    index=False,
)


# Save window-level prediction results
df.to_csv(
    workdir / windows_output,
    index=False,
)


print("\nResults:")
print(result_df.head(20))

print("\nSummary:")
print(f"Number of proteins: {len(result_df)}")
print(f"Mean coverage ratio: {result_df['coverage_ratio'].mean():.4f}")
print(f"Maximum coverage ratio: {result_df['coverage_ratio'].max():.4f}")
print(f"Minimum coverage ratio: {result_df['coverage_ratio'].min():.4f}")

print("\nOutput files:")
print(workdir / coverage_output)
print(workdir / windows_output)