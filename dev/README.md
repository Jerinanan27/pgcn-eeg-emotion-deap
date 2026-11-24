Earlier development versions, kept for transparency. They were not used for any result in the paper.

- `preprocess_eeg.py`, `preprocess_fusion.py`: compute approximate entropy and average Pearson correlation with coherence for the functional edges. The datasets used in the paper were built by `preprocessing/` instead (feature 15 set to zero, absolute-Pearson top-12 edges).
- `pgcn_statistics_plots.py`: the original statistics, which used positive-class F1. Replaced by `analysis/stats_analysis.py`.
