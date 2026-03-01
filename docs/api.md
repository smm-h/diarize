# API Reference

## Main Pipeline

::: diarize.diarize
    options:
      show_root_heading: true
      heading_level: 3

---

## Result Types

::: diarize.DiarizeResult
    options:
      heading_level: 3
      members:
        - segments
        - audio_path
        - audio_duration
        - estimation_details
        - speakers
        - num_speakers
        - to_rttm
        - to_list

::: diarize.Segment
    options:
      heading_level: 3

::: diarize.utils.SpeechSegment
    options:
      heading_level: 3

::: diarize.utils.SubSegment
    options:
      heading_level: 3

::: diarize.SpeakerEstimationDetails
    options:
      heading_level: 3

---

## Voice Activity Detection

::: diarize.vad.run_vad
    options:
      heading_level: 3

---

## Embedding Extraction

::: diarize.embeddings.extract_embeddings
    options:
      heading_level: 3

---

## Clustering

::: diarize.clustering.estimate_speakers
    options:
      heading_level: 3

::: diarize.clustering.cluster_spectral
    options:
      heading_level: 3

::: diarize.clustering.cluster_auto
    options:
      heading_level: 3

::: diarize.clustering.cluster_speakers
    options:
      heading_level: 3

---

## Utilities

::: diarize.utils.get_audio_duration
    options:
      heading_level: 3

::: diarize.utils.format_timestamp
    options:
      heading_level: 3
