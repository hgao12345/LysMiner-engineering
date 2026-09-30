python mpbert_classification.py \
    --config_path config_1024.yaml \
    --load_checkpoint_url LysMiner_Model.ckpt \
    --do_predict True \
    --description classification \
    --num_class 2 \
    --device_id your_device_id \
    --vocab_file vocab_v2.txt \
    --data_url ./example/example_seq.csv \
    --output_url ./example/ \
    --return_sequence False \
    --return_csv True \
    > log.log 2> sys.log
