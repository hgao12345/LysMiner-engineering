import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_name',
    type=str,
    required=True,
    help='Name of the output dataset'
)

parser.add_argument(
    '--pickle_res',
    type=str,
    required=True,
    help='Path to the prediction pickle file generated in Step 4'
)

parser.add_argument(
    '--output_path',
    type=str,
    required=True,
    help='Directory for saving the output CSV and JSON files'
)

args = parser.parse_args()


data_name = args.data_name
pickle_res_path = args.pickle_res
output_path = args.output_path


pickle_res = pickle.load(open(pickle_res_path, "rb"))


select_df = []

select_json = []

select_df.append({
    "id": pickle_res[0]["seq_id"].split("_")[0]+"_WT",
    "seq": "".join(pickle_res[0]["seq"]),
    "mask_logits": 0,
    "label": 0
})

select_json.append({
    "id": pickle_res[0]["seq_id"].split("_")[0]+"_WT",
    "original_seq": "".join(pickle_res[0]["seq"]),
    "pred_seq": "".join(pickle_res[0]["seq"]),
    "truncate_token": pickle_res[0]["seq"],
    "mask_lm_positions": [],
    "logits": []
})

for pred_info in tqdm(pickle_res):
    logits_max = pred_info["logits"].max(axis=1).min()
    if logits_max > 0:
        select_df.append({
            "id": pred_info["seq_id"],
            "seq": "".join(pred_info["pred_seq"]),
            "mask_logits_mean": pred_info["logits"].max(axis=1).mean(),
            "mask_logits_min": pred_info["logits"].max(axis=1).min(),
            "label": -1
        })
        select_json.append({
            "id": pred_info["seq_id"],
            "original_seq": "".join(pred_info["seq"]),
            "pred_seq": "".join(pred_info["pred_seq"]),
            "truncate_token": pred_info["truncate_token"],
            "mask_lm_positions": pred_info["mask_lm_positions"],
            "logits": pred_info["logits"].max(axis=1).tolist()
        })


os.makedirs(output_path, exist_ok=True)

pd.DataFrame(select_df).to_csv(
    os.path.join(output_path, data_name+".csv"),
    index=False
)

json.dump(
    select_json,
    open(os.path.join(output_path, data_name+".json"), "w"),
    indent=4
)

print(len(select_df))

print()
