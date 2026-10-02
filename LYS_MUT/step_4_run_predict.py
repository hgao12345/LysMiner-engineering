import os
from glob import glob
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_name',
    type=str,
    required=True,
    help='Name of the input dataset'
)

parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Path to the input FASTA file'
)

parser.add_argument(
    '--device_id',
    type=str,
    required=True,
    help='Device ID'
)

parser.add_argument(
    '--model_path',
    type=str,
    required=True,
    help='Path to mask_Best_Model.ckpt'
)

parser.add_argument(
    '--save_path',
    type=str,
    required=True,
    help='Directory for saving prediction results'
)

parser.add_argument(
    '--predict_mask_num',
    type=int,
    default=100000,
    help='Number of sequence-generation attempts (default: 100000)'
)

parser.add_argument(
    '--mask_prob',
    type=float,
    default=0.1,
    help='Masking probability (default: 0.1)'
)

args = parser.parse_args()


data_name = args.data_name.replace(".fasta", "")
data_path = args.data_path
device_id = args.device_id
model_path = args.model_path
save_path = args.save_path
predict_mask_num = args.predict_mask_num
mask_prob = args.mask_prob


os.makedirs(save_path,exist_ok=True)
cmd='python ' \
    '/data2/liutuoyu/bert/codes/MP-BERT-v3/mpbert_mask.py ' \
    '--config_path /data2/liutuoyu/bert/codes/MP-BERT-v3/config_1024.yaml ' \
    "--vocab_file  /data2/liutuoyu/bert/codes/generate_mindrecord/generate_for_pretrain_only_Mask/vocab_v2.txt " \
    '--do_predict True ' \
    '--description sequence ' \
    '--device_id '+str(device_id)+' ' \
    '--data_url '+data_path+' ' \
    '--load_checkpoint_url '+model_path+' ' \
    '--output_url '+save_path+' ' \
    '--predict_mask_num '+str(predict_mask_num)+' ' \
    '--mask_prob '+str(mask_prob)+' '
print(cmd,flush=True)
os.system(cmd)
