# Benchmarks

Evaluated on the [VoxConverse](https://github.com/joonson/voxconverse)
dev set (216 files, 1--20 speakers per file).

## Speaker Count Estimation

| GT Speakers | Files | Exact Match | Within +/-1 |
|-------------|-------|-------------|-------------|
| 1 | 22 | 91% | 95% |
| 2 | 44 | 70% | 91% |
| 3 | 35 | 69% | 97% |
| 4 | 24 | 54% | 88% |
| 5 | 31 | 32% | 87% |
| 6--7 | 29 | 45% | 79% |
| 8+ | 31 | 0% | 26% |
| **Overall** | **216** | **51%** | **81%** |

The algorithm works best for 1--4 speakers (88--97% within +/-1).
Accuracy drops for 8 or more speakers --- see
[Limitations](#limitations) below.

## Diarization Error Rate (DER)

DER is the standard metric for speaker diarization, computed with
`collar=0.25` and `skip_overlap=True`.

| System | Weighted DER | Median DER | Notes |
|--------|----------|------------|-------|
| pyannote precision-2 | ~8.5% | -- | Commercial license |
| **diarize** | **~10.8%** | **~3.7%** | **Apache 2.0, CPU-only, no API key** |
| pyannote community-1 | ~11.2% | -- | CC-BY-4.0, needs HF token |
| pyannote 3.1 (legacy) | ~11.2% | -- | MIT, needs HF token |

pyannote DER numbers are self-reported from the
[pyannote benchmark page](https://huggingface.co/pyannote/speaker-diarization-3.1)
on VoxConverse v0.3.

!!! note "Better than pyannote 3.1 on VoxConverse"
    `diarize` achieves lower DER than both pyannote 3.1 (legacy) and
    community-1 on VoxConverse, while requiring no HuggingFace token
    or account registration.

## CPU Speed (Real Time Factor)

RTF = processing_time / audio_duration.  Lower is faster; RTF < 1.0 means
faster than real-time.

| System | Mean RTF | Median RTF | Notes |
|--------|----------|------------|-------|
| **diarize** | **0.12** | **0.12** | **~7x faster than community-1** |
| pyannote community-1 | 0.82 | 0.86 | ~2x faster than 3.1 |
| pyannote 3.1 (legacy) | 1.74 | 1.83 | Slower than real-time on CPU |

Measured on VoxConverse dev files on Apple M2 Pro / M2 Max
(CPU only, no GPU).  All systems were warm-started (models pre-loaded).

!!! note "Apples-to-apples"
    All systems ran on the **same files** with `torch.device("cpu")`.
    `diarize` uses ONNX Runtime for speaker embeddings; pyannote uses
    PyTorch neural networks (segmentation + embedding models).

!!! warning "pyannote 3.1 is slower than real-time on CPU"
    With RTF > 1.0, pyannote 3.1 **cannot process audio in real-time**
    on CPU.  A 10-minute recording takes ~18 minutes to diarize vs
    ~1.2 minutes with `diarize`.  Community-1 is faster (RTF ~0.86)
    but still ~7x slower than `diarize`.

## Methodology

- **Dataset:** VoxConverse dev set --- 216 audio files recorded from
  YouTube debates, news shows, and other multi-speaker media.
- **Ground truth:** RTTM annotations from the
  [official repository](https://github.com/joonson/voxconverse).
- **Evaluation:** [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/)
  `DiarizationErrorRate` with standard parameters.
- **Speed benchmark:** 25 files from VoxConverse dev set, stratified by
  duration.  Wall-clock time measured with `time.time()` after model
  warm-up.  RTF = processing_time / audio_duration.
- **Hardware:** Apple M2 Pro, macOS, CPU only (no GPU).

## Limitations

!!! warning "Speaker count > 7"
    The GMM BIC speaker-count estimator with silhouette refinement works
    well for **1--5 speakers** and degrades gradually for 6--7.  For
    **8 or more speakers** it tends to undercount and produces higher DER.
    If you know your audio has many speakers, pass ``num_speakers``
    explicitly:

    ```python
    result = diarize("panel.wav", num_speakers=12)
    ```

**Known limitations:**

- **Many speakers (8+):** Automatic speaker count estimation degrades ---
  GMM BIC with silhouette refinement reaches 26% within-one accuracy
  for 8+ speakers.  Use ``num_speakers`` when the speaker count is known.
- **Overlapping speech:** DER is computed with ``skip_overlap=True``.
  The pipeline does not model overlapping speech --- when two people
  talk simultaneously, only one is labelled.
- **Short utterances (<&nbsp;0.4 s):** Segments shorter than 0.4 seconds
  are not embedded directly; they are assigned the label of the nearest
  speaker, which can cause errors at speaker boundaries.

## Future Work

!!! info "Single-dataset disclaimer"
    All results above are from VoxConverse dev set only.  We are actively
    expanding evaluation to ensure the algorithm generalises well and is
    not overfit to a single benchmark.

**Planned evaluation:**

- **Cross-dataset validation** --- AMI, DIHARD III, CALLHOME, and other
  standard benchmarks, run in isolated environments with controlled
  CPU/memory limits.
- **Speaker count estimation comparison** --- dedicated benchmarks comparing
  speaker counting accuracy against pyannote and other systems across
  datasets.
- **Broader system comparison** --- benchmark against NeMo, WhisperX, and
  other open-source diarization solutions with verified, reproducible results.

**Planned features:**

- **Streaming / real-time diarization** --- process live audio streams with
  real-time speaker detection and embedding extraction.
- **Speaker identification** --- store and compare speaker embeddings to
  recognise known speakers across sessions.
