import os
from glob import glob
import sys

device_id=sys.argv[1]

data_path=sys.argv[2

]
print(data_path)
cmd='python ' \
    '/data2/liutuoyu/bert/codes/MP-BERT-v3/generate_dataset/generate_seq_for_mask.py ' \
    '--data_dir '+data_path+' ' \
    '--vocab_file /data1/gaohan/lab_bert/MP-BERT-LTY/generate_mindrecord/generate_for_finetune/vocab_v2.txt ' \
    '--output_dir '+data_path+' ' \
    '--max_seq_length 1024 --do_train True --do_eval True --do_test True ' \
    '1> '+data_path+'/data_process_log.log 2> '+data_path+'/data_process_sys.log'
print(cmd,flush=True)
os.system(cmd)

cmd='nohup python ' \
    '/data2/liutuoyu/bert/codes/MP-BERT-v3/mpbert_mask.py ' \
    '--config_path /data2/liutuoyu/bert/codes/MP-BERT-v3/config_1024.yaml ' \
    '--do_train True ' \
    '--do_eval True ' \
    '--description sequence ' \
    '--epoch_num 200 ' \
    '--early_stopping_rounds 50 ' \
    '--frozen_bert False ' \
    '--device_id '+str(device_id)+' ' \
    '--data_url '+data_path+' ' \
    '--load_checkpoint_url /data1/liutuoyu/checkpoint_bert-46210_100.ckpt ' \
    '--output_url '+data_path+' ' \
    '--task_name mask ' \
    '--train_batch_size 32 ' \
    '1> '+data_path+'/train_log.log 2> '+data_path+'/train_sys.log &'
print(cmd,flush=True)
os.system(cmd)
