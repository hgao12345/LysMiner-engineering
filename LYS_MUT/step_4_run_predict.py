#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


# ============================================================
# Parameters
# ============================================================

# Input FASTA file for prediction
data_path = sys.argv[1]

# Device ID
device_id = sys.argv[2]

# Directory containing the trained model
model_dir = (
    "/data4/gh/LysMiner-GITHUB/"
    "LYS_MUT/example/example_seq/train_data"
)

# Trained model
model_path = os.path.join(
    model_dir,
    "mask_Best_Model.ckpt"
)

# Prediction output directory
save_path = os.path.join(
    model_dir,
    "prediction"
)

os.makedirs(
    save_path,
    exist_ok=True
)


# ============================================================
# Check files
# ============================================================

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Input FASTA file not found:\n{data_path}"
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{model_path}"
    )


# ============================================================
# Run prediction
# ============================================================

cmd = (
    "python "
    "mpbert_mask.py "
    
    "--config_path "
    "config_1024.yaml "
    
    "--vocab_file "
    "vocab_v2.txt "
    
    "--do_predict True "
    
    "--description sequence "
    
    "--device_id " + str(device_id) + " "
    
    "--data_url " + data_path + " "
    
    "--load_checkpoint_url " + model_path + " "
    
    "--output_url " + save_path + " "
    
    "--predict_mask_num 100000 "
    
    "--mask_prob 0.1 "
)


# ============================================================
# Print information
# ============================================================

print("=" * 70)
print("LYS-MUT prediction")
print("=" * 70)

print(
    "Input FASTA :",
    data_path
)

print(
    "Model       :",
    model_path
)

print(
    "Output      :",
    save_path
)

print(
    "Device ID   :",
    device_id
)

print("\nCommand:")
print(
    cmd,
    flush=True
)


# ============================================================
# Execute
# ============================================================

return_code = os.system(
    cmd
)

if return_code != 0:

    raise RuntimeError(
        "Prediction failed."
    )


print("\n" + "=" * 70)
print("Prediction completed successfully.")
print("=" * 70)

print(
    "\nResults saved to:",
    save_path
)