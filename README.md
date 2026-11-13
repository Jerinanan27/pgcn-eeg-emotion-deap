# PGCN EEG and facial-video emotion recognition on DEAP

Code for the paper:

**Within and cross-participant evaluation of fusion between EEG and facial video for predicting affect ratings on DEAP**

Jerin Anan Proma, Fatema Tuz Zannat, Mohammed Shawqi Aftab, Shamim H. Ripon
Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh

Release `v1.0-scirep` is the version that corresponds to the paper.

## Repository structure

```
pgcn-eeg-emotion-deap/
├── preprocessing/
│   ├── preprocess_eeg.py          DEAP .dat files -> graph windows (32 participants)
│   └── preprocess_fusion.py       fused_dataset.h5 -> graph windows + video embedding (22 participants)
├── pgcn_core.py                   dataset class, loss, augmentation, evaluation
├── pgcn_model.py                  pyramidal GCN, fusion models and ablation variants
├── pgcn_train.py                  training loops (within-subject and LOSO)
├── pgcn_experiment.py             experiment driver used for every reported run
├── kaggle_notebooks.py            Kaggle cell templates for the training runs
├── analysis/
│   ├── reevaluate_checkpoints.py  re-scores the 408 saved checkpoints (no retraining)
│   ├── stats_analysis.py          Wilcoxon tests, Benjamini-Hochberg correction, effect sizes
│   ├── shuffle_stats.py           statistics for the test-time video manipulation
│   ├── video_shuffle_test.py      test-time video manipulation on the fusion models
│   ├── probe_analysis.py          identity, affect and clip probes; session-drift check
│   ├── check_labels.py            label check against the official DEAP ratings
│   ├── verify_data.py             checks on the saved datasets
│   ├── where_are_missing_videos.py  partition of the trials without video
│   ├── identify_vit.py            optional: identifies the ViT recipe (needs the DEAP videos)
│   ├── identify_vit_from_h5.py    optional: tests candidate ViT weights using only the stored embeddings
│   ├── make_tables.py             all result tables of the paper (LaTeX)
│   └── make_figures.py            result figures of the paper
├── notebooks/
│   ├── preprocessing/             Kaggle notebooks that built the two datasets
│   ├── training/                  Kaggle training notebooks (with outputs)
│   └── revision/                  Kaggle notebook that ran the re-analysis
└── dev/                           earlier development versions, not used for any result
```

The two scripts in `preprocessing/` are the exact code that produced the released datasets, taken from the notebooks in `notebooks/preprocessing/`. They set feature 15 (approximate entropy) to zero and build functional edges from the absolute Pearson correlation (12 strongest per electrode). The files in `dev/` are older versions that compute approximate entropy and coherence-based edges; they were not used for any result.

The trial-level face embeddings in `fused_dataset.h5` were extracted by a co-author with a frozen ViT-B/16 (one frame per second, 60 frames per trial, averaged). That extraction script is not part of this repository; the embeddings are released with the fusion data.

## Data

The DEAP dataset is not redistributed here. Apply for access at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/.

| Data | Link |
|---|---|
| Preprocessed EEG graph data (32 participants) | https://www.kaggle.com/datasets/jarinproma/pgcn-deap-preprocessed |
| Preprocessed fusion data (22 participants) | https://www.kaggle.com/datasets/jarinproma/pgcn-deap-fusion-preprocessed |
| Trained checkpoints and all results | https://www.kaggle.com/datasets/jarinproma/pgcn-all-results |

## Requirements

```
pip install -r requirements.txt
```

PyTorch Geometric depends on your CUDA version: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html. All experiments were run on Kaggle GPUs.

## Reproducing the results

Run every command from the repository root. Scripts in `analysis/` import the model code from the root, so prefix them with `PYTHONPATH=.` (on Kaggle, copy all `.py` files to `/kaggle/working` instead).

**1. Preprocess**

```
python preprocessing/preprocess_eeg.py    --dat_dir <data_preprocessed_python> --out_dir <preprocessed_eeg>
python preprocessing/preprocess_fusion.py --h5_path <fused_dataset.h5>         --out_dir <preprocessed_fusion>
```

**2. Train.** Each configuration (EEG-only, early fusion, late fusion; within-subject and LOSO; ablations `flat_gcn`, `no_band_attn`, `struct_only`, `no_hjorth`) is run with `pgcn_experiment.run_experiment`; see `kaggle_notebooks.py` and `notebooks/training/`. Not every run's notebook was saved, but all runs used `pgcn_experiment.py`.

**3. Re-score the checkpoints** with the training batch sizes (32 within-subject, 16 LOSO), because band attention uses batch statistics:

```
PYTHONPATH=. python analysis/reevaluate_checkpoints.py --data_dir <preprocessed data> --ckpt_dir <run folder> \
    --mode within --fusion none --ablation none --bs 32 --out_prefix reeval_bs/within_none
```

The loop over all 14 configurations is in `notebooks/revision/pgcn_revision_analyses.ipynb`. That notebook also contains earlier copies of some analysis scripts; the versions in `analysis/` are the final ones.

**4. Statistics.** The primary analysis excludes, for each task, participants whose test set lacks a class; the second run keeps everyone (sensitivity analysis).

```
python analysis/stats_analysis.py --in_dir revision_results/reeval_bs --out_dir revision_results/stats_complete --complete_classes
python analysis/stats_analysis.py --in_dir revision_results/reeval_bs --out_dir revision_results/stats_all
```

**5. Probes, video manipulation and checks**

```
python analysis/probe_analysis.py --fus_dir <preprocessed_fusion> --out_dir revision_results/probes --ratings <participant_ratings.csv>
PYTHONPATH=. python analysis/video_shuffle_test.py --fus_dir <preprocessed_fusion> --ckpt_root <results_fusion> --out_dir revision_results/shuffle
python analysis/shuffle_stats.py --shuffle_csv revision_results/shuffle/video_shuffle_per_subject.csv \
    --reeval_dir revision_results/reeval_bs --out revision_results/shuffle/video_shuffle_tests_complete.csv
python analysis/verify_data.py
```

**6. Tables and figures**

```
python analysis/make_tables.py  --res revision_results --out tables
python analysis/make_figures.py --res revision_results --out figures
```

## License

Code: MIT License. DEAP data: subject to the DEAP end-user licence agreement. ViT-B/16 weights: subject to the licence of their source.

## Contact

Corresponding author: Shamim H. Ripon (dshr@ewubd.edu)
Code: Jerin Anan Proma (jarinananproma@gmail.com)
