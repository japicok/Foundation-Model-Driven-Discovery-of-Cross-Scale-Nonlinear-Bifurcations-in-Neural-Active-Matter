# Foundation-Model-Driven Discovery of Cross-Scale Nonlinear Bifurcations in Neural Active Matter

This repository contains the analysis pipeline for detecting and typing fold, Hopf, and pitchfork transitions across the sleep–anesthesia continuum. Five EEG foundation-model embedding streams are aligned on a common basis, projected to a three-dimensional slow manifold, and evaluated with three-scale phase coherence, inter-model disagreement, and embedding-velocity critical exponents.

## Scope

The implemented scientific path includes orthogonal Procrustes alignment, a two-block invertible residual projection, reconstruction and saddle-node normal-form losses, neuronal/population/whole-brain coordinates, the `C ≥ 0.8` coherence gate, the `3σ` disagreement detector, ±120-second log-log critical-exponent fitting, a three-layer bifurcation atlas, subject-disjoint partitions, 10,000-resample bootstrap intervals, Holm–Bonferroni correction, calibration, decision curves, clinical-impact translation, and multi-site heterogeneity.

NeuroLM-XL, LaBraM-base, CBraMod, REVE, and CSBrain remain separate upstream models. Their released token embeddings enter through the common adapter interface. Local spectral, temporal-patch, and convolutional encoders support pipeline validation without representing those upstream weights.

## Installation

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

With conda:

```bash
conda env create -f environment.yml
conda activate neural-bifurcation-atlas
pip install --no-deps -e .
```

With Docker:

```bash
docker build -t neural-bifurcation-atlas .
```

## Data

Verified canonical dataset pages, versions, licenses, and access classes are collected in `dataset_links.txt`. Credentialed PhysioNet and NSRR cohorts require the training and agreements imposed by their custodians. Raw clinical records must remain outside version control.

The nine cohorts are VitalDB, PhysioNet GABAergic multitaper spectra, Sleep-EDF Expanded, SHHS, MESA Sleep Ancillary, CAP Sleep Database, MIMIC-IV, eICU-CRD, and HiRID. The manuscript reports VitalDB, GABAergic spectra, and Sleep-EDF as the primary classification cohorts. ICU databases provide sedation-level and outcome linkage rather than raw scalp EEG.

Prepare an array whose final two dimensions are channel and time:

```bash
nam-prepare --input data/raw/session.npy --output data/windows/session.npy --sample-rate 250 --window-seconds 30 --stride-seconds 2
```

Foundation-model adapters must emit time-aligned, pre-pooling embeddings with a shared row count. Fit Procrustes rotations only on resting-wake reference segments. Never include a subject from a fitted reference, projection, threshold, or atlas partition in its corresponding test partition.

## Training

The manuscript reports released backbone weights, a three-dimensional slow manifold, two hidden invertible layers, `λNF = 0.01`, five models, three biological scales, 15 independent seeds, and an 80/20 subject-disjoint atlas split. It does not report batch size, optimizer, learning rate, epoch count, warmup, decay, clipping, or numeric precision. `configs/main.yaml` labels that status and supplies transparent operational defaults; these values are not manuscript claims.

```bash
nam-train --config configs/main.yaml --embeddings data/derived/vitaldb_aligned.pt --output outputs/vitaldb
```

Override operational values explicitly:

```bash
nam-train --config configs/main.yaml --embeddings data/derived/sleep_edf_aligned.pt training.batch_size=16 training.learning_rate=0.00005
```

The full pipeline is reported to take approximately 48 hours on one NVIDIA A100 80 GB or approximately 14 hours across four A100 GPUs. A 30-second window is reported at 842 ms and 66.9 GB peak memory for the parallel five-model ensemble. These figures include the five upstream backbones and cannot be inferred from the local adapter validation models.

## Evaluation

Prediction archives contain `target`, `probability`, and optionally `baseline_probability` arrays:

```bash
nam-evaluate --predictions outputs/vitaldb/predictions.npz --resamples 10000 --seed 1729
```

Primary expected AUROCs over 15 seeds are 0.937 ± 0.005 on VitalDB, 0.949 ± 0.004 on PhysioNet GABAergic, and 0.946 ± 0.004 on Sleep-EDF. The expected pooled AUROC is 0.944. Scale-count regression values are 0.937, 0.949, and 0.946 for three scales, compared with 0.918, 0.923, and 0.934 for four scales.

Critical-exponent targets are 0.487 for propofol, 0.513 for sevoflurane, 0.478 for isoflurane, 0.632 for dexmedetomidine, and 0.714 for ketamine. Confidence intervals must use session-level resampling when sessions are the independent unit.

## Verification

```bash
pytest -q
ruff check .
mypy --strict code/neural_bifurcation_atlas
```

The test suite covers numerical invariants, signal transforms, subject disjointness, invertibility, losses, calibration, bootstrap comparisons, multiple-testing correction, heterogeneity, critical-exponent recovery, and an optimization regression.

## Method boundaries

Scalp high-gamma is a mesoscopic population surrogate and does not resolve single-unit firing. The whole-brain coordinate is a cortically summed mean-field assumption for recordings with at least 16 channels. The detector is retrospective research software and does not provide clinical recommendations.

The manuscript does not give the numeric normal-coordinate tolerance `η`, coherence window `W`, disagreement baseline window `W`, reference-cohort construction details, precise backbone adapter layer names, or the full atlas label-generation protocol. These inputs must be fixed in an analysis plan before reporting new results. Defaults in the configuration make software behavior explicit but are not substitutes for missing experimental metadata.

## License

The source is released under the MIT License. Dataset terms remain controlled by each data custodian, and some datasets prohibit redistribution or require credentialing.
