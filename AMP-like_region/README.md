# AMP-like Region Prediction

This module identifies AMP-like regions within full-length protein sequences using a sliding-window-based prediction strategy.

Each protein sequence is divided into overlapping 13-amino-acid windows with a step size of 1. Each window is then classified using the pretrained AMP classification model **ABP-MPB**. Windows predicted as AMP-positive (`pred_label = 1`) are mapped back to their original positions in the protein sequence, and the proportion of residues covered by AMP-positive windows is calculated for each protein.

The workflow consists of three steps:

1. Generate overlapping 13-aa sliding-window sequences.
2. Classify each window using the pretrained ABP-MPB model.
3. Calculate the proportion of each protein covered by AMP-positive windows.

## Requirements

Prepare an environment with:

- MindSpore 1.8.0
- Python packages:
  - `numpy`
  - `pandas`
  - `tqdm`

The prediction step also requires the following MP-BERT/ABP-MPB files:

```text
mpbert_classification.py
config_1024.yaml
vocab_v2.txt
```

These files should be placed in the working directory from which `02-classification-predict.py` is executed.

## Input Data

Protein sequences should be provided in CSV format with two columns:

| Column | Description |
| :-- | :-- |
| `id` | Protein identifier |
| `seq` | Protein sequence |

For example:

```csv
id,seq
protein_1,MKTIIALSYIFCLVFADYKDDDDK
protein_2,MKWVTFISLLLLFSSAYSRGVFRR
```

Protein identifiers should be unique. Input protein sequences should contain at least 13 amino acid residues.

## Pretrained ABP-MPB Model

The pretrained **ABP-MPB** model checkpoint is required for AMP-like region prediction and must be downloaded separately before running the prediction step.

The pretrained model is available from [Zenodo](https://zenodo.org/records/16545412).

After downloading the checkpoint, specify its local path using the `--load_checkpoint_url` argument in Step 2.

## Step 1. Generate Sliding-Window Sequences

Use `01-sliding-windows.py` to divide each full-length protein sequence into overlapping 13-aa windows with a step size of 1.

For example:

```bash
python 01-sliding-windows.py \
    --work_path ./example \
    --seq_name example_seq
```

The main arguments are:

| Argument | Description | Default |
| :-- | :-- | :-- |
| `--work_path` | Working directory containing the input CSV file | Required |
| `--seq_name` | Name of the input CSV file without the `.csv` extension | Required |

For example, with:

```text
--work_path ./example
--seq_name example_seq
```

the script reads:

```text
./example/example_seq.csv
```

and generates:

```text
./example/example_seq_sliding_window_sequences13AA.csv
```

Each generated window is assigned an ID consisting of the original protein identifier followed by its 1-based starting position.

For example:

```text
protein_1_1
protein_1_2
protein_1_3
...
```

The sliding-window size and step size are fixed at 13 and 1, respectively.

## Step 2. Predict AMP-Positive Windows

Use `02-classification-predict.py` to classify the 13-aa windows using the pretrained ABP-MPB model.

For example:

```bash
python 02-classification-predict.py \
    --device_id 0 \
    --data_dir ./example/example_seq_sliding_window_sequences13AA.csv \
    --output_dir ./example/AMP_prediction \
    --load_checkpoint_url /path/to/ABP-MPB_checkpoint.ckpt
```

The main arguments are:

| Argument | Description | Default |
| :-- | :-- | :-- |
| `--device_id` | Device ID used for prediction | Required |
| `--data_dir` | Path to the sliding-window CSV file generated in Step 1 | Required |
| `--output_dir` | Directory for saving prediction results and logs | Required |
| `--load_checkpoint_url` | Path to the downloaded pretrained ABP-MPB checkpoint | Required |

The script uses:

```text
mpbert_classification.py
config_1024.yaml
vocab_v2.txt
```

for model prediction.

Prediction logs are saved under:

```text
output_dir/
└── logs/
    ├── log.log
    └── sys.log
```

The prediction output contains the predicted class (`pred_label`) for each 13-aa window. In this workflow:

```text
pred_label = 1  ->  AMP-positive
pred_label = 0  ->  AMP-negative
```

The prediction result CSV generated in this step is used as the input for Step 3.

## Step 3. Calculate AMP-like Region Coverage

Use `03-ratio-cal.py` to map AMP-positive windows back to their original protein positions and calculate the proportion of each protein covered by AMP-positive windows.

For example:

```bash
python 03-ratio-cal.py \
    --workdir ./example/AMP_prediction \
    --input_file example_seq_sliding_window_sequences13AA_predict_result.csv
```

The main arguments are:

| Argument | Description | Default |
| :-- | :-- | :-- |
| `--workdir` | Working directory containing the prediction result CSV | Required |
| `--input_file` | Prediction result CSV generated in Step 2 | Required |
| `--window_size` | Sliding-window size used for coverage calculation | `13` |

The default window size is 13 amino acids, consistent with the sliding-window generation strategy used in Step 1 and should normally remain unchanged.

For each protein, all residues covered by AMP-positive windows (`pred_label = 1`) are combined. Overlapping residues are counted only once when calculating the AMP-like region coverage.

The output filenames are generated automatically from the input prediction filename.

For example, if the input file is:

```text
example_seq_sliding_window_sequences13AA_predict_result.csv
```

the script generates:

```text
example_seq_sliding_window_sequences13AA_predict_result_protein_coverage_ratios.csv
example_seq_sliding_window_sequences13AA_predict_result_all_windows.csv
```

The main protein-level output file contains:

| Column | Description |
| :-- | :-- |
| `protein_name` | Original protein identifier |
| `protein_length` | Protein length inferred from the sliding-window positions |
| `positive_fragments` | Number of 13-aa windows predicted as AMP-positive |
| `unique_positive_positions` | Number of unique residues covered by AMP-positive windows |
| `coverage_ratio` | Proportion of the protein sequence covered by AMP-positive windows |

The AMP-like region coverage ratio is calculated as:

```text
coverage_ratio = unique_positive_positions / protein_length
```

The second output file contains the window-level prediction results together with the extracted protein name, window starting position, and AMP-positive status.

## Example Workflow

A complete example workflow is:

```bash
# Step 1: Generate 13-aa sliding windows
python 01-sliding-windows.py \
    --work_path ./example \
    --seq_name example_seq

# Step 2: Predict AMP-positive windows
python 02-classification-predict.py \
    --device_id 0 \
    --data_dir ./example/example_seq_sliding_window_sequences13AA.csv \
    --output_dir ./example/AMP_prediction \
    --load_checkpoint_url /path/to/ABP-MPB_checkpoint.ckpt

# Step 3: Calculate AMP-like region coverage
python 03-ratio-cal.py \
    --workdir ./example/AMP_prediction \
    --input_file example_seq_sliding_window_sequences13AA_predict_result.csv
```

The resulting `coverage_ratio` represents the fraction of residues in each full-length protein covered by 13-aa windows classified as AMP-positive by ABP-MPB.
