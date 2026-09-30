# AMP-like Region Prediction

This module identifies AMP-like regions within full-length protein sequences using a sliding-window-based prediction strategy.

Each protein sequence is divided into overlapping 13-amino-acid windows with a step size of 1. Each window is then classified using the pretrained AMP classification model **ABP-MPB**. Windows predicted as AMP-positive (`pred_label = 1`) are mapped back to their original positions in the protein sequence and merged to identify continuous AMP-like regions. The proportion of residues covered by AMP-like regions is subsequently calculated for each protein.

## Input Data

Protein sequences should be provided in CSV format with two columns:

| id | seq |
| :-- | :-- |
| Protein identifier | Protein sequence |

For example:

```csv
id,seq
protein_1,MKTIIALSYIFCLVFADYKDDDDK
protein_2,MKWVTFISLLLLFSSAYSRGVFRR
```

Protein identifiers should be unique. Input protein sequences should be at least 13 amino acids long.

An example input file is provided at:

```text
example/example_seq.csv
```

## Pretrained AMP Model

The pretrained **ABP-MPB** model used for AMP-like region prediction can be downloaded from Zenodo:

https://zenodo.org/records/16545412

Download `ABP_Model.ckpt` and save it to a local directory. The checkpoint path can then be specified using the `--checkpoint` argument.

## Usage

To run the pipeline with the example dataset:

```bash
python AMP_region_prediction.py \
    --checkpoint /path/to/ABP_Model.ckpt
```

By default, the pipeline reads:

```text
example/example_seq.csv
```

and writes the results to the `example/` directory.

For a custom protein dataset:

```bash
python AMP_region_prediction.py \
    --input /path/to/input.csv \
    --checkpoint /path/to/ABP_Model.ckpt \
    --output_dir /path/to/output \
    --device 0
```

### Main Arguments

| Argument | Description | Default |
| :-- | :-- | :-- |
| `--input` | Input CSV file containing protein IDs and sequences | `example/example_seq.csv` |
| `--checkpoint` | Path to the pretrained `ABP_Model.ckpt` | Required |
| `--output_dir` | Directory for output files | `example/` |
| `--device` | GPU/device ID used for prediction | `0` |
| `--seq_name` | Prefix used for sliding-window output files | `LYS` |

The sliding-window size and step size are fixed at 13 and 1, respectively, consistent with the model construction and AMP-like region prediction strategy used in this study.

## Output

The pipeline generates the following files:

```text
output_directory/
├── LYS_sliding_window_sequences13AA.csv
├── protein_AMP_region_ratio.csv
└── AMP_prediction/
    ├── LYS_sliding_window_sequences13AA_predict_result.csv
    └── logs/
        ├── log.log
        └── sys.log
```

The main output file is:

```text
protein_AMP_region_ratio.csv
```

It contains the following information:

| Column | Description |
| :-- | :-- |
| `protein_name` | Protein identifier |
| `protein_length` | Length of the protein sequence |
| `total_windows` | Total number of 13-aa windows generated |
| `positive_windows` | Number of windows predicted as AMP-positive |
| `AMP_length` | Number of residues covered by AMP-like regions |
| `AMP_ratio` | Proportion of the protein sequence covered by AMP-like regions |
| `AMP_percent` | AMP-like region coverage expressed as a percentage |
| `AMP_regions` | Positions of continuous AMP-like regions in the protein sequence |

For example:

```text
protein_name,protein_length,total_windows,positive_windows,AMP_length,AMP_ratio,AMP_percent,AMP_regions
protein_1,150,138,25,42,0.28,28.0,15-36;82-101
```

Here, `AMP_regions = 15-36;82-101` indicates two predicted AMP-like regions spanning residues 15–36 and 82–101.

## Workflow

The complete pipeline consists of three steps:

1. **Sliding-window generation** – full-length protein sequences are divided into 13-aa windows with a step size of 1.
2. **AMP prediction** – each window is classified using the pretrained ABP-MPB model, with `pred_label = 1` considered AMP-positive.
3. **AMP-like region calculation** – overlapping AMP-positive windows are merged and mapped back to the full-length protein to calculate AMP-like region coverage.
