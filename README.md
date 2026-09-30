# LysMiner-engineering

This repository provides a workflow for lysozyme identification, AMP-like region prediction, and lysozyme activity engineering using three modules: **LysMiner**, **AMP-like region prediction**, and **LYS-MUT**.

## Environment Setup

Follow the [DLFea4AMPGen installation guide](https://github.com/hgao12345/DLFea4AMPGen#installation) to configure the MindSpore environment. Its reference setup uses Linux with a Huawei Ascend 910 NPU, Docker ≥18.03, Python 3.7.5, and MindSpore Ascend 1.8.0.

After installing MindSpore and the matching hardware runtime, install the core Python dependencies using versions from the [reference environment](https://github.com/hgao12345/DLFea4AMPGen/blob/main/requirement.txt):

```bash
python -m pip install numpy==1.21.6 pandas==1.0.5 scikit-learn==0.24.1 PyYAML==6.0 six==1.16.0 tqdm==4.66.4
```

LYS-MUT additionally requires HMMER 3.1b2 (`jackhmmer`), a local UniRef90 FASTA database, and the complete MP-BERT masked-model components described in its README.

## Choose a Workflow

| Task | Module | Instructions |
| :-- | :-- | :-- |
| Identify candidate lysozymes | LysMiner | [LysMiner README](LysMiner/README.md) |
| Calculate AMP-like region proportions | AMP-like | [AMP-like README](AMP-like_region/README.md) |
| Generate and screen variants for activity engineering | LYS-MUT → LysMiner → AMP-like | [LYS-MUT README](LYS_MUT/README.md) |

### Lysozyme Identification

Use **LysMiner** with a CSV file containing `id` and `seq` columns. The model returns a predicted class and a lysozyme-positive score. Sequences with `pred_label == 1` are retained as candidate lysozymes.

### AMP-like Region Prediction

Use **AMP-like** with a CSV file containing `id` and `seq` columns. The module scans each protein using 13-residue windows with a one-residue step, merges positive regions, and calculates the fraction of the protein sequence covered by these regions. Overlapping residues are counted only once.

### Lysozyme Activity Engineering

1. Provide one wild-type protein sequence in FASTA format.
2. Use **LYS-MUT** to retrieve homologous sequences, train a target-specific model, and generate candidate variants. The checkpoint `mask_Best_Model.ckpt` is produced during training.
3. Screen the generated variants with **LysMiner** and retain only lysozyme-positive sequences.
4. Use **AMP-like** to calculate the AMP-like region proportion of each retained variant.
5. Rank the variants by AMP-like region proportion in descending order and select the top candidates for synthesis and experimental validation.

LysMiner and AMP-like prediction require their respective trained classification checkpoints; see the module READMEs for model preparation and execution commands.

LysMiner predicts lysozyme identity rather than quantitative activity in U/mg. AMP-like region proportion is a candidate-selection criterion; increased enzymatic activity must be confirmed experimentally.
