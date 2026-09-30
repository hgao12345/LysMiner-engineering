# AMP-like Region Prediction

This module predicts intrinsic AMP-like regions in protein sequences and calculates the proportion of residues covered by these regions. Each protein is scanned using a 13-amino-acid sliding window with a step size of one residue. A single trained AMP prediction model classifies each window, and overlapping or adjacent positive windows are merged into continuous AMP-like regions.

First, prepare a Python environment with MindSpore and the following packages:

```bash
python -m pip install numpy "pandas<2" scikit-learn pyyaml six tqdm
```

The study used MindSpore 1.8.0. Install the corresponding MindSpore package and hardware dependencies for your platform; see the [MindSpore 1.8 installation documentation](https://www.mindspore.cn/docs/zh-CN/r1.8/faq/installation.html). The supplied `config_1024.yaml` uses `device_target: "Ascend"` by default. Ensure that this setting matches your installed MindSpore backend and available hardware. The current classification script uses a legacy pandas API, so pandas 1.x is required unless that call is updated.

Then, organize the input data into the following format:

```text
AMP-like_region/
└── example/
    └── example_seq.csv
```

The CSV file must contain a header and the following columns in this order:

| id | seq |
| :--: | :--: |
| Unique protein ID | Full-length amino acid sequence |

Use uppercase amino acid sequences without spaces or gaps. Each sequence must contain at least 13 residues. A `label` column is not required for prediction, and the input CSV does not need to be converted to MindRecord format.

After that, update the corresponding parameters in `AMP-like_region_prediction.py`:

```python
PROJECT_DIR = Path("/absolute/path/to/AMP-like_region")
WORK_DIR = PROJECT_DIR / "example"
INPUT_FILE = WORK_DIR / "example_seq.csv"

CHECKPOINT = Path("/absolute/path/to/test_Best_Model.ckpt")
DEVICE_ID = 0
```

`PROJECT_DIR` must contain `mpbert_classification.py`, `config_1024.yaml`, `vocab_v2.txt`, and the `src/` directory. `CHECKPOINT` must point to a trained AMP classification checkpoint compatible with this model configuration. The checkpoint is not included in this directory and must be provided separately.

Keep the following settings to reproduce the supplied scanning procedure:

```python
SEQ_NAME = "LYS"
WINDOW_SIZE = 13
STEP_SIZE = 1
POSITIVE_LABEL = 1
```

`SEQ_NAME` controls the intermediate output filenames. `DEVICE_ID` selects the Ascend device when using the default backend. Keep `STEP_SIZE = 1`, because the current coverage calculation infers the protein length from the final sliding window.

Then, run the prediction pipeline from the project directory:

```bash
cd /absolute/path/to/AMP-like_region
python AMP-like_region_prediction.py
```

The script automatically generates sliding windows, predicts each window using one checkpoint, and calculates the AMP-like region coverage of each protein. Windows with `pred_label == 1` are treated as AMP-positive.

With the default filenames, the results are saved in the following format:

```text
example/
├── example_seq.csv
├── LYS_sliding_window_sequences13AA.csv
├── AMP_prediction/
│   ├── LYS_sliding_window_sequences13AA_predict_result.csv
│   └── logs/
│       ├── log.log
│       └── sys.log
└── protein_AMP_region_ratio.csv
```

The final output, `protein_AMP_region_ratio.csv`, contains the following columns:

| Column | Description |
| :-- | :-- |
| `protein_name` | Protein ID from the input file |
| `protein_length` | Protein sequence length |
| `total_windows` | Number of sliding windows evaluated |
| `positive_windows` | Number of windows predicted as AMP-positive |
| `AMP_length` | Number of unique residues covered by positive windows |
| `AMP_ratio` | `AMP_length / protein_length`, ranging from 0 to 1 |
| `AMP_percent` | `AMP_ratio × 100`, expressed as a percentage |
| `AMP_regions` | Merged AMP-like regions using 1-based, inclusive coordinates, separated by semicolons |

Results are sorted by `AMP_ratio` in descending order. Overlapping residues are counted only once. Proteins with no positive windows receive an `AMP_ratio` of zero and an empty `AMP_regions` field.

AMP-like region coverage describes predicted sequence features; it does not directly measure antimicrobial potency or lysozyme enzymatic activity. If prediction fails, check `example/AMP_prediction/logs/log.log` and `sys.log` for details.
```


