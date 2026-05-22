# my_KWS — context for Claude

This file is read automatically by Claude Code at the start of any session in this repo. It exists so the project context survives across machines / new conversations.

---

## What this project is

A personal "Hey computer" keyword-spotting (KWS) wake-word detector, built from scratch as a learning project. End goal: record-your-own wake-word data, train a small classifier, run live mic inference, then push the model through quantization → MATLAB fixed-point → an automated validation framework → edge-deployment simulation.

The user is following a 6-chunk learning roadmap calibrated to a Bose-style edge-audio DSP/ML role.

## Sibling reference repo

`../edgeaudio-ml/` is a reference implementation **for Chunk 1 only** (data + features). Treat it as a peek-after-you-try reference, not a template to copy. The user is deliberately rebuilding each layer themselves in `my_KWS` to internalize the concepts (边学边写). Do **not** suggest cloning files from `edgeaudio-ml` into `my_KWS`.

Key difference: `edgeaudio-ml` does **12-class** Speech Commands KWS. `my_KWS` is **2-class** ("hey computer" vs not). Speech Commands keywords and ESC-50 clips are negatives/background for this project, not separate classes.

---

## 6-Chunk roadmap

### Chunk 1 — Data Foundation (~1 week) — **IN PROGRESS**
- Combine Google Speech Commands (KWS) + ESC-50 (scenes) as negative/background pool.
- Record user's own "hey computer" samples across acoustic conditions (quiet / music / walking / whispered / far).
- Feature extraction in Python: MFCCs, log-mel spectrograms, framing/windowing.
- **Deliverable:** reproducible `src/my_kws/` module (io, features, augment, datasets) + EDA notebook (class distributions, SNR analysis, spectrogram visualizations).
- **Learn:** audio I/O, framing, windowing, mel filterbanks, MFCC math, augmentation (noise mixing, time-shift, SpecAugment).

### Chunk 2 — Float32 Baseline Models (~1 week)
- Tiny CNN for KWS (~20–50K params, <100KB target).
- Targets: >90% KWS accuracy on held-out set.
- **Deliverable:** training scripts, checkpoints, evaluation report.
- **Learn:** small-model design, accuracy/size tradeoffs, TC-ResNet / MatchboxNet / depthwise separable convs.

### Chunk 3 — Quantization & Optimization (~1.5 weeks)
- Float32 → post-training int8 → QAT → manual fixed-point reference.
- Compare accuracy, latency, model size at each step.
- **Learn:** int8 scale/zero-point math, why QAT recovers accuracy, which layers resist quantization.

### Chunk 4 — MATLAB Fixed-Point Simulation (~1.5 weeks)
- Re-implement KWS inference path in Q15/Q31 using MATLAB Fixed-Point Designer.
- Front-end DSP (pre-emphasis, framing, FFT, mel filterbank, log) in fixed-point.
- Bit-exact compare against Python float reference; tune per-layer word lengths.
- **Learn:** Q-format, saturation vs wrap, fixed-point FFT, Python↔MATLAB/C bridging.

### Chunk 5 — Automated Validation Framework (~1.5 weeks) — **portfolio centerpiece**
- Python framework: takes `(model, toolchain_config)`, runs battery of tests (accuracy, edge-case noisy/quiet/accented, latency, memory footprint, bit-exactness vs reference).
- Emits HTML/Markdown report with pass/fail + regression detection vs baseline.
- Runs in CI (GitHub Actions) on every commit.
- **Deliverable:** `validate.py` CLI + sample reports + CI badge.
- **Learn:** real ML engineering — directly mirrors the Bose JD bullet on "automation of a system validating the impact of toolchain updates."

### Chunk 6 — Qualcomm-Style Deployment Simulation (~1 week)
- Profile memory layout for a Kalimba-like architecture (separate program/data memory, MIPS budget).
- TFLite Micro or microTVM → C codegen; run on laptop with artificial CPU/memory throttling.
- Writeup: "if I had a QCC5181, here's the port plan and what I'd measure."
- **Learn:** edge deployment realities, model-architecture × hardware interactions, reading DSP reference manuals.

---

## Current state (2026-05-21)

- **Chunk:** 1
- **Step in Chunk 1:** Step 1 (project skeleton + `uv sync`) — in progress, `pyproject.toml` started.
- **Next action:** finish `pyproject.toml` (add dependencies + build-system + hatch wheel target), create directory skeleton, run `uv sync`, verify imports.

### Step 1–6 breakdown for Chunk 1

| Step | Build | Learn | Reference (peek after) |
|---|---|---|---|
| 1 | Project skeleton (`pyproject.toml`, `src/my_kws/__init__.py`, `data/`, `scripts/`, `notebooks/`, `tests/`) + `uv sync` | Python package layout, uv workflow | `edgeaudio-ml/pyproject.toml` |
| 2 | `scripts/download_data.py` — pull Speech Commands v2 + ESC-50 into `data/raw/` | Dataset structure, why non-keyword + background data matters | `edgeaudio-ml/scripts/download_data.py` |
| 3 | `src/my_kws/io.py` — `load_wav` + resample + pad/crop + RMS/SNR helpers | Sample rate, mono, float32 range, RMS math | `edgeaudio-ml/src/edgeaudio/io.py` |
| 4 | `src/my_kws/features.py` — **numpy-from-scratch** framing/windowing/STFT/mel/log first, then torchaudio wrapper | DSP soul: Hamming window, STFT, mel scale, log compression, MFCC = log-mel + DCT | `edgeaudio-ml/src/edgeaudio/features.py` |
| 5 | `src/my_kws/augment.py` — `add_noise` (by SNR), `random_time_shift`, `spec_augment` (time/freq mask) | SNR formula, why augmentation helps generalization, SpecAugment as cutout on spectrograms | `edgeaudio-ml/src/edgeaudio/augment.py` |
| 6 | Adapt `scripts/record_wakewords.py` for "hey computer" + `src/my_kws/datasets.py` (binary `HeyComputerDataset`) + EDA notebook (class distributions, SNR histograms, log-mel examples) | PyTorch Dataset protocol, why binary KWS needs lots of negatives, EDA storytelling | `edgeaudio-ml/src/edgeaudio/datasets.py`, `edgeaudio-ml/scripts/record_wakewords.py` |

---

## Working style — important

- **User prefers 边学边写**: break work into small steps, explain the relevant DSP/ML concept at the point it's needed. Don't dump a finished implementation. Let the user write code first, then code-review against the reference.
- **Communication**: user writes prompts in Chinese, comfortable with English code/comments. Mix Chinese narration with English technical terms (mel-spectrogram, STFT, MFCC, etc.) — don't over-translate jargon.
- **Don't add features the user didn't ask for**. Bug fixes don't need cleanup, one-shot operations don't need helpers.
- **For UI/notebooks**: user chose Jupyter notebooks (not `scripts/eda.py` + `reports/`) for the EDA deliverable. Inline plots + audio widgets + markdown narration in one file.

## Conventions

- Python 3.10–3.12, managed by `uv`.
- `numpy < 2.0` (librosa/torchaudio compat).
- Sample rate: 16 kHz. Clip length: TBD (likely 1.5–2.0s for "hey computer" — confirm from EDA of own recordings before fixing).
- Feature defaults (likely; mirror `edgeaudio-ml`): n_fft=512, win_length=400, hop_length=160, n_mels=40, output shape (40, 98) for 1s.

## Don't do

- Don't fork or copy files from `edgeaudio-ml` wholesale. The point is rebuilding them.
- Don't suggest 12-class KWS architectures or losses — this project is binary.
- Don't skip Chunk 1 EDA to rush into modeling. Class imbalance (positives ≪ negatives) and SNR distribution mismatches are real risks that EDA exists to surface.
