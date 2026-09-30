# AMP-like Region Prediction

This module is used to identify AMP-like regions within full-length protein sequences using a sliding-window-based prediction strategy.

The workflow first divides each protein sequence into 13-amino-acid overlapping windows with a step size of 1. Each window is then classified using a trained AMP prediction model. Windows predicted as AMP-positive (`label = 1`) are merged according to their positions in the original protein sequence to identify continuous AMP-like regions and calculate their proportion within each protein.

## Input Data

First, prepare the protein sequences in CSV format:

> input.csv

The input CSV file should contain the following columns:

| id | seq |
| :--: | :--: |
| protein id | protein sequence |

For example:

```csv
id,seq
protein_1,MKTIIALSYIFCLVFADYKDDDDK
protein_2,MKWVTFISLLLLFSSAYSRGVFRR
```

## AMP-like Region Prediction

Run the prediction pipeline using:

```bash
python AMP_region_prediction.py
```

The main parameters can be specified at the beginning of the script:

```python
WORK_DIR = Path("<working_directory>")
INPUT_FILE = WORK_DIR / "<input_file>.csv"

WINDOW_SIZE = 13
STEP_SIZE = 1

DEVICE_ID = <device_id>
POSITIVE_LABEL = 1
```

### Pretrained AMP Model

The pretrained AMP classification model (ABP-MPB) used for AMP-like region prediction can be downloaded from [Zenodo](https://zenodo.org/records/16545412/files/ABP_Model.ckpt?download=1).

After downloading the model, specify the path to `ABP_Model.ckpt` in the script:

```python
CHECKPOINT = Path("<path_to>/ABP_Model.ckpt")
```

The paths to the MP-BERT model script, configuration file, and vocabulary file should also be specified:

```python
MODEL_SCRIPT = Path("<path_to>/mpbert_classification.py")
CONFIG_PATH = Path("<path_to>/config_1024.yaml")
VOCAB_FILE = Path("<path_to>/vocab_v2.txt")
```

## Workflow

The pipeline consists of three steps.

### 1. Generate Sliding Windows

Each full-length protein sequence is divided into 13-aa overlapping windows using a step size of 1.

For a protein sequence of length *L*, a total of:

```text
L - 13 + 1
```

windows are generated.

The resulting file is organized as:

| id | seq |
| :--: | :--: |
| protein_1_1 | 13-aa sequence |
| protein_1_2 | 13-aa sequence |
| ... | ... |

The number at the end of each ID indicates the starting position of the window in the original protein sequence.

### 2. Predict AMP-like Windows

Each 13-aa window is classified using the pretrained ABP-MPB model.

In this pipeline:

```text
pred_label = 1
```

is defined as an AMP-positive window.

Only a single trained model is used for prediction; no 10-fold ensemble is required.

### 3. Identify AMP-like Regions

AMP-positive windows are mapped back to their original positions in the full-length protein.

Overlapping positive windows are merged into continuous AMP-like regions.

The AMP-like region proportion is calculated as:

```text
AMP-like region proportion =
number of residues covered by AMP-positive windows / full protein length
```

## Output

The final output file is:

```text
protein_AMP_region_ratio.csv
```

The output contains the following information:

| Column | Description |
| :-- | :-- |
| protein_name | Protein ID |
| protein_length | Length of the full-length protein |
| total_windows | Total number of 13-aa windows |
| positive_windows | Number of AMP-positive windows |
| AMP_length | Number of residues covered by AMP-like regions |
| AMP_ratio | Fraction of the protein covered by AMP-like regions |
| AMP_percent | Percentage of the protein covered by AMP-like regions |
| AMP_regions | Positions of continuous AMP-like regions |

For example:

```text
protein_name,protein_length,total_windows,positive_windows,AMP_length,AMP_ratio,AMP_percent,AMP_regions
protein_1,150,138,20,35,0.2333,23.33,18-45;109-115
```

## Requirements

The pipeline requires:

```text
Python
pandas
NumPy
MP-BERT
MindSpore
```

The MP-BERT configuration file, vocabulary file, and pretrained ABP-MPB checkpoint are required for prediction.
