# LYS-MUT Training and Prediction

LYS-MUT is a sequence generation framework used to construct target-specific lysozyme variant libraries. Starting from a wild-type lysozyme, homologous sequences are retrieved from UniRef90 to construct a target-specific training dataset. The resulting model is then used to generate sequence variants of the input lysozyme.

The target-specific checkpoint `mask_Best_Model.ckpt` is generated during model training.

## Requirements

Prepare an environment with:

- MindSpore 1.8.0
- HMMER 3.1b2 (`jackhmmer`)
- Python packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `pyyaml`
  - `six`
  - `tqdm`

The default hardware backend used in this study is Ascend.

A local copy of the UniRef90 protein sequence database is also required for homologous sequence retrieval.

## 1. Prepare the Input Sequence

Save one wild-type lysozyme sequence in FASTA format, for example:

```text
example/example_seq.fasta
```

The input FASTA file must contain exactly one protein sequence:

```fasta
>example_seq
MKTIIALSYIFCLVFADYKDDDDK...
```

Use uppercase amino acid letters without spaces or gaps.

With the default model configuration, the input sequence should contain 10–1022 amino acid residues.

## 2. Retrieve Homologous Sequences and Prepare the Dataset

Use `step_1_run_sequence_align.py` to search a local UniRef90 database with `jackhmmer` and construct the training, validation, and test datasets.

For example:

```bash
python step_1_run_sequence_align.py \
    --fasta_file ./example/example_seq.fasta \
    --uniref /path/to/uniref90.fasta \
    --output ./example \
    --threshold 1.0 \
    --aggregation 1 \
    --cpu 8
```

Replace `/path/to/uniref90.fasta` with the path to your local UniRef90 FASTA database.

The main arguments are:

| Argument | Description | Default |
| :-- | :-- | :-- |
| `-F`, `--fasta_file` | Input FASTA containing exactly one protein sequence | Required |
| `-U`, `--uniref` | Path to the local UniRef90 FASTA database | Required |
| `-O`, `--output` | Output directory | Directory containing the input FASTA |
| `-T`, `--threshold` | HMMER score-threshold coefficient relative to query sequence length | `0.5` |
| `-A`, `--aggregation` | Training-data aggregation factor | `1` |
| `--cpu` | Number of CPU threads used by `jackhmmer` | `8` |

For the LYS-MUT workflow used in this study, `--threshold 1.0` and `--aggregation 1` were used.

The script performs a `jackhmmer` search with five iterations and generates the following files:

```text
example/
├── example_seq.hmmer.out.o
├── example_seq.hmmer.tblout
├── example_seq.hmmer.domtblout
├── example_seq.hmmer.fasta
│
└── example_seq/
    └── train_data/
        ├── train.fasta
        ├── val.fasta
        └── test.fasta
```

The retrieved sequences are randomly divided into training, validation, and test sets using a fixed random seed (`random_state = 42`) at an approximate ratio of:

```text
Train : 64%
Val   : 16%
Test  : 20%
```

## 3. Configure Model Training and Prediction

Before training, configure the paths used by the following scripts:

| Script | Setting |
| :-- | :-- |
| `step_3_run_train.py` | Set `data_path` to the path of `example/example_seq/train_data/` |
| `step_4_run_predict.py` | Set `model_dir` to the same training directory |
| `step_5_get_result.py` | Set `PREDICTION_DIR` to the `prediction/` subdirectory generated under the training directory |

The training and prediction wrappers also require the MP-BERT masked-model implementation and its associated files, including the model configuration, `vocab_v2.txt`, `generate_seq_for_mask.py`, and a pretrained initialization checkpoint.

The pretrained initialization checkpoint used to initialize model training is distinct from the target-specific `mask_Best_Model.ckpt` generated during LYS-MUT training.

## 4. Train the Target-Specific Model

After generating the homologous-sequence dataset, convert the datasets to the required MindRecord format and train the model using:

```bash
python step_3_run_train.py 0
```

Replace `0` with the appropriate Ascend device ID.

Training runs in the background. Progress and error messages are written to:

```text
train_log.log
train_sys.log
```

After training and evaluation are completed, the target-specific checkpoint:

```text
mask_Best_Model.ckpt
```

is saved in the training directory.

## 5. Generate Lysozyme Variants

After training is complete, generate candidate variants from the wild-type lysozyme using:

```bash
python step_4_run_predict.py \
    ./example/example_seq.fasta \
    0
```

Replace `0` with the appropriate Ascend device ID.

By default, the generation procedure performs 100,000 sequence-generation attempts with a masking proportion of 0.1.

These settings can be adjusted in `step_4_run_predict.py` through:

```text
--predict_mask_num
--mask_prob
```

The generated sequences and associated model outputs are written to the `prediction/` directory under the corresponding training directory.

## 6. Export Generated Variants

Finally, convert the prediction outputs into CSV and JSON files:

```bash
python step_5_get_result.py example_seq
```

The argument should correspond to the input FASTA filename without its extension.

For the example above, the final results are saved under:

```text
example/example_seq/train_data/prediction/
```

including:

```text
example_seq.csv
example_seq.json
```

The CSV file contains the following columns:

| Column | Description |
| :-- | :-- |
| `id` | Sequence identifier |
| `seq` | Protein sequence |
| `mask_logits_mean` | Mean masked-position model score |
| `mask_logits_min` | Minimum masked-position model score |
| `label` | Sequence type indicator |

In the output, `label = 0` denotes the wild-type sequence and `label = -1` denotes generated candidate variants. These labels indicate sequence origin and should not be interpreted as lysozyme activity labels.
