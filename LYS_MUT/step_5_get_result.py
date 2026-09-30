#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import json
import sys

import pandas as pd
from tqdm import tqdm


# ============================================================
# Parameters
# ============================================================

# Usage:
# python select_prediction.py example_seq

data_name = sys.argv[1]

# Remove extension if accidentally provided.
data_name = data_name.replace(".fasta", "")
data_name = data_name.replace(".pkl", "")
data_name = data_name.replace(".csv", "")
data_name = data_name.replace(".json", "")


# ============================================================
# Prediction directory
# ============================================================

PREDICTION_DIR = (
    "/data4/gh/LysMiner-GITHUB/"
    "LYS_MUT/example/example_seq/train_data/prediction"
)


# ============================================================
# Input / output files
# ============================================================

PICKLE_FILE = os.path.join(
    PREDICTION_DIR,
    data_name + ".pkl"
)

CSV_FILE = os.path.join(
    PREDICTION_DIR,
    data_name + ".csv"
)

JSON_FILE = os.path.join(
    PREDICTION_DIR,
    data_name + ".json"
)


# ============================================================
# Check input file
# ============================================================

if not os.path.exists(PICKLE_FILE):

    raise FileNotFoundError(
        "Prediction pickle file not found:\n"
        f"{PICKLE_FILE}"
    )


# ============================================================
# Load prediction results
# ============================================================

print("=" * 70)
print("Process LYS-MUT prediction results")
print("=" * 70)

print(
    "Input pickle:",
    PICKLE_FILE
)

pickle_res = pickle.load(
    open(
        PICKLE_FILE,
        "rb"
    )
)

print(
    "Total prediction records:",
    len(pickle_res)
)


# ============================================================
# Prepare output
# ============================================================

select_df = []

select_json = []


# ============================================================
# Add WT sequence
# ============================================================

wt_id = (
    pickle_res[0]["seq_id"]
    .split("_")[0]
    + "_WT"
)

wt_seq = "".join(
    pickle_res[0]["seq"]
)


select_df.append({

    "id": wt_id,

    "seq": wt_seq,

    "mask_logits_mean": 0,

    "mask_logits_min": 0,

    "label": 0
})


select_json.append({

    "id": wt_id,

    "original_seq": wt_seq,

    "pred_seq": wt_seq,

    "truncate_token":
        pickle_res[0]["seq"],

    "mask_lm_positions": [],

    "logits": []
})


# ============================================================
# Select generated sequences
# ============================================================

for pred_info in tqdm(
    pickle_res,
    desc="Processing predictions"
):

    # Maximum prediction score at each mutated position.
    position_logits = (
        pred_info["logits"]
        .max(axis=1)
    )

    # Minimum score among all mutated positions.
    logits_max = (
        position_logits.min()
    )

    # Keep sequences for which all selected positions
    # have a positive prediction score.
    if logits_max > 0:

        select_df.append({

            "id":
                pred_info["seq_id"],

            "seq":
                "".join(
                    pred_info["pred_seq"]
                ),

            "mask_logits_mean":
                position_logits.mean(),

            "mask_logits_min":
                position_logits.min(),

            "label":
                -1
        })


        select_json.append({

            "id":
                pred_info["seq_id"],

            "original_seq":
                "".join(
                    pred_info["seq"]
                ),

            "pred_seq":
                "".join(
                    pred_info["pred_seq"]
                ),

            "truncate_token":
                pred_info[
                    "truncate_token"
                ],

            "mask_lm_positions":
                pred_info[
                    "mask_lm_positions"
                ],

            "logits":
                position_logits.tolist()
        })


# ============================================================
# Save CSV
# ============================================================

result_df = pd.DataFrame(
    select_df
)

result_df.to_csv(
    CSV_FILE,
    index=False
)


# ============================================================
# Save JSON
# ============================================================

with open(
    JSON_FILE,
    "w"
) as f:

    json.dump(
        select_json,
        f,
        indent=4
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("Completed")
print("=" * 70)

print(
    "Selected sequences:",
    len(select_df)
)

print(
    "CSV output:",
    CSV_FILE
)

print(
    "JSON output:",
    JSON_FILE
)