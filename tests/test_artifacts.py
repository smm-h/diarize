"""Tests for opt-in pipeline artifacts (``diarize(..., return_artifacts=True)``)."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from diarize import DiarizeArtifacts, DiarizeResult, diarize
from diarize.utils import SpeakerEstimationDetails, SpeechSegment, SubSegment


def _run_pipeline(**kwargs):
    """Run diarize() with mocked VAD, embeddings, and clustering."""
    speech = [SpeechSegment(start=0.0, end=3.0), SpeechSegment(start=4.0, end=7.0)]
    subs = [
        SubSegment(start=0.0, end=1.2, parent_idx=0),
        SubSegment(start=4.0, end=5.2, parent_idx=1),
    ]
    embeddings = np.random.default_rng(0).standard_normal((2, 256))
    labels = np.array([0, 1])
    details = SpeakerEstimationDetails(method="gmm_bic", best_k=2, pca_dim=8)

    with (
        patch("diarize.get_audio_duration", return_value=10.0),
        patch("diarize.run_vad", return_value=speech),
        patch("diarize.extract_embeddings", return_value=(embeddings, subs)),
        patch("diarize.cluster_speakers", return_value=(labels, details)),
    ):
        result = diarize("test.wav", **kwargs)
    return result, embeddings, subs


class TestArtifacts:
    def test_artifacts_none_by_default(self):
        result, _, _ = _run_pipeline()

        assert result.artifacts is None
        assert result.model_dump()["artifacts"] is None

    def test_return_artifacts_populates_embeddings_and_subsegments(self):
        result, embeddings, subs = _run_pipeline(return_artifacts=True)

        assert isinstance(result.artifacts, DiarizeArtifacts)
        assert result.artifacts.embeddings == embeddings.tolist()
        assert result.artifacts.subsegments == subs
        # Plain Python types, not numpy arrays.
        assert isinstance(result.artifacts.embeddings, list)
        assert all(isinstance(row, list) for row in result.artifacts.embeddings)
        assert len(result.artifacts.embeddings[0]) == 256

    def test_populated_artifacts_serialize_to_json(self):
        result, embeddings, subs = _run_pipeline(return_artifacts=True)

        data = json.loads(result.model_dump_json())
        assert data["artifacts"]["embeddings"] == embeddings.tolist()
        assert data["artifacts"]["subsegments"] == [s.model_dump() for s in subs]

        restored = DiarizeResult.model_validate_json(result.model_dump_json())
        assert restored == result
