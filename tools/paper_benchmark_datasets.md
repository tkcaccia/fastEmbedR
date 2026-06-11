# Paper Benchmark Datasets

This inventory collects the datasets explicitly used in the TriMap paper
(`1910.00204v2`) and the LocalMAP paper (`2412.15426v2`) so fastEmbedR can
build a broad, reproducible benchmark without relying on private datasets,
Kaggle credentials, or manual browser downloads.

The machine-readable inventory is in `tools/paper_benchmark_datasets.csv`.

## Recommended benchmark tiers

### Tier 0: smoke tests

Use these in CI and during local development.

| Dataset | Paper | Size in paper | Source plan |
| --- | --- | ---: | --- |
| S-curve | TriMap | 5,000 x 3 | Generate with `sklearn.datasets.make_s_curve`. |
| COIL-20 small | TriMap, LocalMAP | 1,440 x 16,384 | Columbia CAVE COIL-20 processed images. |

### Tier 1: core publication benchmark

These are feasible on a laptop or workstation and cover image, text, tabular,
and biological data.

| Dataset | Paper | Size in paper | Source plan |
| --- | --- | ---: | --- |
| MNIST | TriMap, LocalMAP | 70,000 x 784 | Official MNIST files, torchvision, or OpenML `mnist_784`. |
| Fashion-MNIST | TriMap, LocalMAP | 70,000 x 784 | Zalando Research GitHub/direct files or torchvision. |
| USPS | TriMap, LocalMAP | 9,298 x 256 | OpenML `USPS` by default; optional Kaggle replication uses `bistaumanga/usps-dataset`. |
| 20 Newsgroups | TriMap, LocalMAP | 18,846 x 100 | `sklearn.datasets.fetch_20newsgroups`; TF-IDF then SVD to 100 dimensions. |
| Epileptic Seizure | TriMap | 11,500 x 178 | UCI Epileptic Seizure Recognition. |
| Kang PBMC | LocalMAP | 13,999 x 1,000 | Figshare/GEO GSE96583, log-normalize, select 1,000 HVGs. |
| Seurat/BMCITE | LocalMAP | 30,672 x 1,000 | Likely Stuart 2019 bone marrow CITE-seq via SeuratData `bmcite`; verify exact labels. |
| Tabula Muris | TriMap | about 54,000 x 1,000 | Tabula Muris public files, log-normalize, select 1,000 HVGs. |

### Tier 2: large scalability benchmark

Use these for the R Journal performance tables and GPU/remote machine runs.

| Dataset | Paper | Size in paper | Source plan |
| --- | --- | ---: | --- |
| TV News | TriMap | about 129,000 | UCI TV News Channel Commercial Detection. |
| Covertype | TriMap | about 581,000 x 54 | UCI or `sklearn.datasets.fetch_covtype`. |
| RCV1 | TriMap | about 800,000 | `sklearn.datasets.fetch_rcv1`, sparse text benchmark. |
| Character Font Images | TriMap | paper says about 1.7M; current UCI page reports 745,000 | UCI Character Font Images; use current source size and record version. |
| Human Cortex | LocalMAP | 43,349 x 1,000 | Allen Brain Map / public human cortex data; verify exact Zhu 2023 source. |
| CBMC | LocalMAP | 67,686 x 1,000 | Ambiguous: paper cites Stoeckius 2017, but common public CBMC CITE-seq is 8,617 cells. Use only after exact source is confirmed. |

### Tier 3: huge optional stress tests

These should not be part of ordinary CI. Run on the remote GPU workstation or
as explicitly requested long benchmarks.

| Dataset | Paper | Size in paper | Source plan |
| --- | --- | ---: | --- |
| KDDCup99 | TriMap | about 4.9M | `sklearn.datasets.fetch_kddcup99` or UCI. |
| HIGGS | TriMap | 11M x 28 | UCI HIGGS direct files. |
| PBMC 1M | LocalMAP | 1,263,676 x 1,000 | Perez 2022 lupus PBMC, CELLxGENE/GEO/public H5AD where available. |
| AIDA | LocalMAP | 1,058,909 x 1,000 | AIDA Data Freeze v1, likely through CELLxGENE or project release. |

## Optional Kaggle replication

These datasets are not part of the default benchmark because they require
Kaggle authentication. They can be included for paper-replication runs after
accepting Kaggle terms and installing the Kaggle CLI.

| Dataset | Paper | Kaggle slug | Benchmark name |
| --- | --- | --- | --- |
| USPS | TriMap, LocalMAP | `bistaumanga/usps-dataset` | `kaggle_usps` |
| 360K+ Lyrics | TriMap | `gyani95/380000-lyrics-from-metrolyrics` | `kaggle_lyrics` |

Prepare the optional Kaggle datasets with:

```bash
python3 -m pip install --user kaggle h5py
python3 tools/prepare_kaggle_paper_datasets.py \
  --dataset all \
  --cache-dir results/rjournal_benchmark/cache/kaggle
```

Your current-token path `~/.kaggle/access_token` is supported by the current
Kaggle CLI. The legacy `~/.kaggle/kaggle.json` path is also supported.

Then run a benchmark that explicitly requests them:

```bash
Rscript tools/rjournal_benchmark.R \
  --datasets=kaggle_usps,kaggle_lyrics \
  --download-kaggle \
  --max-n=5000 \
  --k=15,30
```

For full paper-scale Lyrics replication, omit the row cap and allow enough disk,
RAM, and time for TF-IDF/SVD preprocessing.

For a quick Lyrics loader smoke test, prepare a small file and point the R
benchmark at it:

```bash
python3 tools/prepare_kaggle_paper_datasets.py \
  --dataset lyrics \
  --cache-dir results/rjournal_benchmark/cache/kaggle \
  --lyrics-max-rows 5000

FASTEMBEDR_KAGGLE_LYRICS_FILE=kaggle_metrolyrics_svd100_n5000.npz \
Rscript tools/rjournal_benchmark.R \
  --datasets=kaggle_lyrics \
  --cache-dir=results/rjournal_benchmark/cache \
  --max-n=2000 \
  --k=15
```

## Special handling

| Dataset | Paper | Reason |
| --- | --- | --- |
| CIFAR-10 CNN features | TriMap | The paper embeds features from a trained CNN layer, not raw CIFAR-10. Include only after adding a deterministic feature-extraction pipeline. |

## Suggested metrics by paper

TriMap-style benchmark:

- runtime and peak memory
- nearest-neighbour preservation
- global distance/cluster placement metrics
- very large scalability runs

LocalMAP-style benchmark:

- silhouette by labels
- post-hoc classification accuracy
- stability across seeds/initializations
- biological label preservation for single-cell data

For fastEmbedR, keep all individual metrics visible. A combined score can be
reported, but it should never hide the speed/quality trade-off.
