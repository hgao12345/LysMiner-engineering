# LysMiner Prediction

LysMiner identifies candidate lysozymes from protein sequences using a trained binary classification model. For each input sequence, it returns a predicted class and a score for the lysozyme-positive class.

The study used MindSpore 1.8.0. Install the corresponding MindSpore package and hardware dependencies for your platform; see the [MindSpore 1.8 installation documentation](https://www.mindspore.cn/docs/zh-CN/r1.8/faq/installation.html). The supplied configuration uses `device_target: "Ascend"` by default. Ensure that this setting matches your installed MindSpore backend and available hardware. The current classification script uses a legacy pandas API, so pandas 1.x is required unless that call is updated.

Then, organize the input data into the following format:

```text
LysMiner/
└── example/
    └── example_seq.csv
```

The CSV file must contain a header and the following columns:

| id | seq |
| :--: | :--: |
| Unique protein ID | Protein amino acid sequence |

Use uppercase amino acid sequences without spaces or gaps. A `label` column is not required for prediction, and the input CSV does not need to be converted to MindRecord format. With the supplied configuration, sequences longer than 1021 residues are truncated to their first 1021 residues before prediction.

After that, place the trained LysMiner checkpoint in the project directory:

```text
LysMiner/
├── LysMiner_Best_Model.ckpt
├── LysMiner_prediction.sh
├── mpbert_classification.py
├── config_1024.yaml
├── vocab_v2.txt
├── src/
└── example/
    └── example_seq.csv
```

Download the trained [LysMiner checkpoint](https://zenodo.org/records/23055127) and place it in the project directory. Alternatively, set `--load_checkpoint_url` to the actual path of the downloaded checkpoint.

Then, run the following command from the project directory:

```bash
cd /absolute/path/to/LysMiner
python mpbert_classification.py \
    --config_path ./config_1024.yaml \
    --load_checkpoint_url ./LysMiner_Best_Model.ckpt \
    --do_predict True \
    --description classification \
    --num_class 2 \
    --device_id 0 \
    --vocab_file ./vocab_v2.txt \
    --data_url ./example/example_seq.csv \
    --output_url ./example/ \
    --return_sequence False \
    --return_csv True \
    1> log.log 2> sys.log
```

Replace `--device_id 0` with the ID of your available Ascend device. Change `--data_url` and `--output_url` to use a different input file or output directory. The output directory must exist before running the command.

Alternatively, edit the paths and device ID in `LysMiner_prediction.sh`, then run:

```bash
bash LysMiner_prediction.sh
```

The supplied shell script uses device ID `6`. Its relative `--config_path config_1024.yaml` is resolved under `src/model_utils/`; use `--config_path "$PWD/config_1024.yaml"` as shown above to select the top-level configuration explicitly.

For the example input, the prediction results are saved to `example/example_seq_predict_result.csv` with the following columns:

| Column | Description |
| :-- | :-- |
| `id` | Protein ID from the input file |
| `seq` | Original input sequence, including any residues excluded by truncation |
| `pred_label` | Predicted class: `1` for candidate lysozymes and `0` for non-lysozymes |
| `dense` | Model softmax score for class `1`, ranging from 0 to 1 |

The CSV also contains a leading row-index column written by pandas. This column can be ignored. `pred_label` is determined by the class with the highest score; the output rows retain the input order.

To retain only the predicted lysozyme-positive sequences for subsequent analysis, run the following command from the project directory:

```bash
python - <<'PY'
import pandas as pd

results = pd.read_csv("example/example_seq_predict_result.csv")
positive = results.loc[results["pred_label"] == 1, ["id", "seq"]]
positive.to_csv("example/lysozyme_positive.csv", index=False)
PY
```

The resulting `lysozyme_positive.csv` can be used as input for AMP-like region prediction. LysMiner predicts lysozyme identity; its classification score does not quantify enzymatic activity in U/mg. Experimental validation is required to confirm activity. If prediction fails, check `log.log` and `sys.log` in the project directory.

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


