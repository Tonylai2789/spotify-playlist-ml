# Spotify Playlist Clustering and GRU Recommendation

An ML class final project exploring playlist-level audio clustering and a GRU-based next-track candidate scorer.

## Start here

- [Jupyter notebook: code excerpts and explanations](spotify_playlist_ml.ipynb)
- [Original full project report (19 pages)](reports/Spotify_ML_Project_Report.pdf)

**Archive status:** The notebook packages the four code blocks available in the May 2025 report, plus an explicitly labeled import cell and explanatory Markdown. It is not the recovered full training notebook or an independently reproduced experiment. No notebook cells were executed for this publication.

## Project approach

1. Combine existing Spotify Million Playlist track occurrences with a SQLite song/audio-feature lookup.
2. Summarize each playlist with the means and standard deviations of nine audio features (18 features total).
3. Apply PCA and search MiniBatchKMeans configurations to group playlists.
4. Use cluster-informed negative examples for a GRU-based scorer of playlist context and candidate tracks, as described in the report.
5. Evaluate the reported clustering and candidate-scoring results.

Original playlist order supplies observed next-track positives. K-means clusters are not ground-truth labels; sampled negatives are not necessarily bad musical recommendations.

## Included and missing

Included code: playlist aggregation, PCA, MiniBatchKMeans search, the original commented-out KMeans alternative, and the GRU prediction call. The import cell and explanatory notes are packaging additions. Misleading comments were corrected without changing the underlying computations; the notebook explains these edits.

Not available in the report: data loaders/SQL join, original preprocessing, train/test preparation, negative sampler, complete GRU model/training, Optuna objectives, GPU implementation, or saved weights. These have not been reconstructed or invented. The notebook therefore requires external variables such as `full_df`, `model`, `X_seq_te`, and `X_cand_te` and is **not runnable end to end as supplied**.

No raw datasets, API keys, personal account credentials, or pretrained model files are included. The exact SQLite dataset source and usage terms must be recovered before redistributing data.

## Reported results

Historical values transcribed from the PDF, not rerun benchmarks:

| Experiment | Result |
|---|---|
| Baseline / tuned clustering silhouette | 0.2821 / 0.2939 |
| Baseline / tuned Davies-Bouldin | 1.0424 / 1.0220 |
| Selected cluster count | 4 |
| GRU at 50,000 playlists, ROC-AUC | 0.708 |
| GRU at 50,000 playlists, report's MAP column | 0.507 |

ROC-AUC is not next-song accuracy. The report's AP/MAP terminology needs the original metric code to resolve. Optuna results were mixed across sample sizes. The report does not prove that cluster-informed negative sampling improves over random sampling; a controlled comparison is a future experiment.

## Viewing and environment

GitHub renders the notebook without running it. No CUDA or TensorFlow installation is required to view it or the PDF. A Jupyter environment is needed only to open it interactively locally.

The shown clustering snippets use Python, NumPy, pandas, and scikit-learn (`n_init='auto'` requires a supporting version). The full experiment also discusses a GRU, GPU acceleration, and Optuna, but the original environment and dependency versions were not preserved. No dependency lockfile or GPU setup is asserted here.

## Credits and provenance

The original report is titled **Spotify ML project Report**, dated **May 2025**, and credits **Tony Lai and Ian Jang**. The PDF is preserved unchanged from the original Overleaf project. This repository does not change its authorship or results.

Notebook and README annotations were prepared for archival clarity in September 2026. They flag unresolved standardization, sampling, and evaluation details rather than treating report prose as implementation proof. The original PDF includes historical wording that is not independently verified by this archive.
