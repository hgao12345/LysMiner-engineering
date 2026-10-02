import os
from glob import glob
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--device_id',
    type=str,
    required=True,
    help='Device ID'
)

parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Path to the training data'
)

parser.add_argument(
    '--load_checkpoint_url',
    type=str,
    required=True,
    help='Path to the pretrained checkpoint'
)

args = parser.parse_args()

device_id = args.device_id
data_path = args.data_path
load_checkpoint_url = args.load_checkpoint_url


cmd='python ' \
    'generate_seq_for_mask.py ' \
    '--data_dir '+data_path+' ' \
    '--vocab_file vocab_v2.txt ' \
    '--output_dir '+data_path+' ' \
    '--max_seq_length 1024 --do_train True --do_eval True --do_test True ' \
    '1> '+data_path+'/data_process_log.log 2> '+data_path+'/data_process_sys.log'
print(cmd,flush=True)
os.system(cmd)

cmd='nohup python ' \
    'mpbert_mask.py ' \
    '--config_path config_1024.yaml ' \
    '--do_train True ' \
    '--do_eval True ' \
    '--description sequence ' \
    '--epoch_num 200 ' \
    '--early_stopping_rounds 50 ' \
    '--frozen_bert False ' \
    '--device_id '+str(device_id)+' ' \
    '--data_url '+data_path+' ' \
    '--load_checkpoint_url '+load_checkpoint_url+' ' \
    '--output_url '+data_path+' ' \
    '--task_name mask ' \
    '--train_batch_size 32 ' \
    '1> '+data_path+'/train_log.log 2> '+data_path+'/train_sys.log &'
print(cmd,flush=True)
os.system(cmd)
