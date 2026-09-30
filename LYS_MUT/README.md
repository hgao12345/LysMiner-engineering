# LYS-MUT Training and Prediction

LYS-MUT trains a target-specific model using homologous sequences and generates candidate variants from a wild-type lysozyme. The checkpoint `mask_Best_Model.ckpt` is generated during training.

Prepare an environment with MindSpore 1.8.0, HMMER 3.1b2 (`jackhmmer`), and the Python packages `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `six`, and `tqdm`. The default hardware backend is Ascend.

First, save one wild-type protein sequence in `example/example_seq.fasta`. Use uppercase amino acid letters without spaces or gaps. With the default configuration, the sequence must contain 10–1022 residues.

Then, search your local UniRef90 database and prepare the training data:

```bash
cd /absolute/path/to/LYS_MUT
python step_1_run_sequence_align.py \
    -F "$PWD/example/example_seq.fasta" \
    -U /absolute/path/to/uniref90.fasta \
    -O "$PWD/example" \
    -T 1.0 -A 1
```

Replace `-U` with your database path. This creates `train.fasta`, `val.fasta`, and `test.fasta` under `example/example_seq/train_data/`. The script requests 50 CPU threads; adjust its `--cpu` setting if needed.

Before training, update the server-specific paths in these scripts:

| Script | Setting |
| :-- | :-- |
| `step_3_run_train.py` | Set `data_path` to the absolute path of `example/example_seq/train_data/` |
| `step_4_run_predict.py` | Set `model_dir` to the same training directory |
| `step_5_get_result.py` | Set `PREDICTION_DIR` to that directory's `prediction/` subfolder |

The wrappers also require paths to a complete MP-BERT masked-model implementation, its matching configuration, `vocab_v2.txt`, `generate_seq_for_mask.py`, and a pretrained initialization checkpoint. The current copy lacks some of these components; provide them or point the wrappers to a complete installation. The initialization checkpoint is separate from the target-specific checkpoint produced below.

Next, convert the datasets to MindRecord format and train the model:

```bash
python step_3_run_train.py 0
```

Replace `0` with your Ascend device ID. Training runs in the background and saves `mask_Best_Model.ckpt` in the training directory. Check `train_log.log` and `train_sys.log`, and wait for training and evaluation to finish.

Then, generate variants using the trained model:

```bash
python step_4_run_predict.py "$PWD/example/example_seq.fasta" 0
```

The default settings are 100,000 generation attempts and a masking proportion of 0.1. Change `--predict_mask_num` and `--mask_prob` inside the script if needed.

Finally, export the results:

```bash
python step_5_get_result.py example_seq
```

Pass the input filename without its extension. Results are saved in `example/example_seq/train_data/prediction/` as `example_seq.csv` and `example_seq.json`. The CSV includes `id`, `seq`, `mask_logits_mean`, `mask_logits_min`, and `label`; `label` marks the wild type (`0`) or generated candidates (`-1`), not lysozyme activity.

Use the candidate `id` and `seq` columns for subsequent LysMiner screening and AMP-like region prediction.
```
