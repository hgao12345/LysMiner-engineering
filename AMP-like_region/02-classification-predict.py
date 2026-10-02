import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--device_id',
    type=str,
    required=True,
    help='Device ID'
)

parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Path to the input CSV file'
)

parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Directory for saving prediction results'
)

parser.add_argument(
    '--load_checkpoint_url',
    type=str,
    required=True,
    help='Path to the trained model checkpoint'
)

args = parser.parse_args()


device_id = args.device_id
data_dir = args.data_dir
output_dir = args.output_dir
load_checkpoint_url = args.load_checkpoint_url


os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)


cmd = 'python ' \
    'mpbert_classification.py ' \
    '--config_path config_1024.yaml ' \
    '--load_checkpoint_url '+load_checkpoint_url+' ' \
    '--do_predict True ' \
    '--description classification ' \
    '--num_class 2 ' \
    '--device_id '+str(device_id)+' ' \
    '--vocab_file vocab_v2.txt ' \
    '--data_url '+data_dir+' ' \
    '--output_url '+output_dir+' ' \
    '--return_sequence False ' \
    '--return_csv True ' \
    '1> '+os.path.join(output_dir, 'logs/log.log')+' ' \
    '2> '+os.path.join(output_dir, 'logs/sys.log')

print(cmd, flush=True)
os.system(cmd)