# PGCN EEG Emotion Recognition on DEAP

Code for the paper:

**"Trial-Level Evaluation of EEG–Facial-Video Fusion for Emotion Recognition on DEAP: Within-Subject Gains and Cross-Subject Limitations"**

Jerin Anan Proma, Fatema Tuz Zannat, Shamim H. Ripon  
Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh  

---

## Overview

This repository contains the complete code for preprocessing, model training, fusion experiments, ablation study, ViT embedding probes, and corrected statistical analysis.

---

## Repository Structure

```
pgcn-eeg-emotion-deap/
├── preprocess_eeg.py           # EEG .dat → graph windows (32 subjects)
├── preprocess_fusion.py        # Fused H5 → EEG + ViT video graphs
├── pgcn_core.py                # Dataset, loss, augmentation, evaluation
├── pgcn_model.py               # PyramidalGCN + all ablation variants
├── pgcn_train.py               # Training loops (within-subject and LOSO)
├── pgcn_experiment.py          # Experiment driver used for all results
├── pgcn_statistics_plots.py    # Figures and original statistics
├── corrected_stats.py          # Corrected Wilcoxon + BH + rank-biserial
├── kaggle_notebooks.py         # Kaggle cell templates for all experiments
└── README.md
```

---

## Data

**DEAP dataset** — not redistributed here. Request access from the original distributors:  
https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

**Preprocessed data and all results** — publicly available on Kaggle:

| Dataset | URL |
|---|---|
| Preprocessed EEG (32 subjects) | https://www.kaggle.com/datasets/jarinproma/pgcn-deap-preprocessed |
| Preprocessed Fusion (22 subjects) | https://www.kaggle.com/datasets/jarinproma/pgcn-deap-fusion-preprocessed |
| All experiment results | https://www.kaggle.com/datasets/jarinproma/pgcn-all-results |

> Update the Kaggle URLs above with your exact public dataset links before submission.

---

## Requirements

```bash
pip install torch torch-geometric scipy numpy pandas scikit-learn h5py tqdm
```

PyTorch Geometric installation depends on your CUDA version:  
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

All experiments were run on Kaggle (NVIDIA Tesla T4, Python 3.12).

---

## Reproducing the Results

### Step 1 — Preprocess EEG

```bash
python preprocess_eeg.py \
    --dat_dir /path/to/data_preprocessed_python \
    --out_dir /path/to/preprocessed_eeg
```

### Step 2 — Preprocess Fusion

```bash
python preprocess_fusion.py \
    --h5_path /path/to/fused_dataset.h5 \
    --out_dir /path/to/preprocessed_fusion
```

### Step 3 — Run Experiments

See `kaggle_notebooks.py` for the exact Kaggle workflow used across four accounts in parallel. The six main configurations are:

| Config | Mode | Fusion |
|---|---|---|
| EEG-Only | within / LOSO | none |
| Early Fusion | within / LOSO | early |
| Late Fusion | within / LOSO | late |

Ablation variants: `flat_gcn`, `no_band_attn`, `struct_only`, `no_hjorth`

### Step 4 — ViT Embedding Probes

Run on your preprocessed fusion dataset (22 real-video subjects). Requires `scikit-learn`, `numpy`, `torch`. See Methods Section 3 of the paper for the exact probe code.

### Step 5 — Statistics

Run in a Kaggle notebook (paste as a cell, not from command line):

```python
RESULTS = "/kaggle/input/your-results-dataset/all_results"
OUT     = "/kaggle/working/statistics_all_comparisons.csv"
```

Then run `stats.ipynb` with those paths set at the top of the file. Outputs a CSV with raw p-values, Holm-corrected p-values, Benjamini–Hochberg-corrected p-values, and matched-pairs rank-biserial effect sizes for all comparisons. These are the numbers reported in Tables 4, 5, and 6 of the paper.

---


## License

Code: MIT License.  
DEAP dataset: subject to the original DEAP end-user license agreement.  
Pre-trained ViT-B/16 weights: subject to the license of the source repository used for feature extraction.

---

## Contact

Jerin Anan Proma — jerinanan27@gmail.com  
East West University, Department of CSE, Dhaka, Bangladesh
