"""Behavioural tests for the Bayesian-surprise segmenter.

Run:
    PYTHONPATH=src python src/emllm/tests/test_segmenter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emllm.segmenter import BayesianSurpriseSegmenter, SegmenterConfig


VOCAB = 1024


def _stable_logits(seed: int) -> torch.Tensor:
    """Logits that put almost all mass on a single token, for stable streams."""
    g = torch.Generator().manual_seed(seed)
    base = torch.full((VOCAB,), -10.0)
    idx = int(torch.randint(0, VOCAB, (1,), generator=g).item())
    base[idx] = 10.0
    return base


def test_no_boundary_on_repeated_distribution() -> None:
    """Identical logits -> KL is 0 -> never crosses any threshold."""
    cfg = SegmenterConfig(window=16, min_segment_len=8, max_segment_len=10_000)
    seg = BayesianSurpriseSegmenter(cfg)
    logits = _stable_logits(0)
    boundaries = []
    for _ in range(100):
        b = seg.step(logits)
        if b is not None:
            boundaries.append(b)
    assert boundaries == [], f"expected no boundaries, got {len(boundaries)}"


def test_distribution_shift_creates_boundary() -> None:
    """A single sharp shift in the predicted token should create a boundary."""
    cfg = SegmenterConfig(window=32, min_segment_len=8, max_segment_len=10_000,
                          threshold_quantile=0.85)
    seg = BayesianSurpriseSegmenter(cfg)
    a = _stable_logits(0)
    b = _stable_logits(7)
    boundaries = []
    # 30 stable steps on `a`, one switch to `b`, 30 stable steps on `b`
    for _ in range(30):
        out = seg.step(a)
        if out:
            boundaries.append(out)
    out = seg.step(b)
    if out:
        boundaries.append(out)
    for _ in range(30):
        out = seg.step(b)
        if out:
            boundaries.append(out)
    assert len(boundaries) >= 1, "expected at least one boundary across the shift"
    # The first boundary should be on or shortly after the shift
    first = boundaries[0]
    assert 28 <= first.position <= 35, f"boundary at {first.position}, expected ~30"


def test_max_segment_len_force_cuts() -> None:
    """When no surprise spikes occur, max_segment_len triggers a cut."""
    cfg = SegmenterConfig(window=8, min_segment_len=4, max_segment_len=20)
    seg = BayesianSurpriseSegmenter(cfg)
    logits = _stable_logits(0)
    boundaries = []
    for _ in range(50):
        b = seg.step(logits)
        if b is not None:
            boundaries.append(b)
    # Expect ~50/20 = 2 forced cuts
    assert 2 <= len(boundaries) <= 3, f"got {len(boundaries)} forced cuts"


def test_close_emits_final_segment() -> None:
    cfg = SegmenterConfig(window=8, min_segment_len=4)
    seg = BayesianSurpriseSegmenter(cfg)
    for _ in range(20):
        seg.step(_stable_logits(0))
    final = seg.close()
    assert final.position == 20
    segs = list(seg.segments())
    assert segs[-1] == (0, 20), f"final segment {segs[-1]} != (0, 20)"


def test_segments_iterator_covers_stream_without_overlap() -> None:
    cfg = SegmenterConfig(window=16, min_segment_len=4, max_segment_len=12)
    seg = BayesianSurpriseSegmenter(cfg)
    for _ in range(40):
        seg.step(_stable_logits(0))
    seg.close()
    segs = list(seg.segments())
    # Contiguous coverage and end-exclusive
    assert segs[0][0] == 0
    for (a_start, a_end), (b_start, b_end) in zip(segs, segs[1:]):
        assert a_end == b_start, f"gap or overlap between {(a_start, a_end)} and {(b_start, b_end)}"


if __name__ == "__main__":
    tests = [
        ("no_boundary_on_repeated_distribution", test_no_boundary_on_repeated_distribution),
        ("distribution_shift_creates_boundary", test_distribution_shift_creates_boundary),
        ("max_segment_len_force_cuts", test_max_segment_len_force_cuts),
        ("close_emits_final_segment", test_close_emits_final_segment),
        ("segments_iterator_covers_stream_without_overlap",
         test_segments_iterator_covers_stream_without_overlap),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
