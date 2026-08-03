"""Tests for my_kws.streaming — detector logic without any model.

The StreamingDetector takes score_fn as an injected dependency, so the
trigger / refractory / consecutive logic is testable with scripted scores.
"""

import numpy as np
import pytest

from my_kws.streaming import HOP, SR, WINDOW, StreamingDetector


def make_detector(scores, **kwargs):
    """Detector whose score_fn returns scripted values in sequence."""
    it = iter(scores)
    last = {"v": 0.0}

    def score_fn(_x):
        last["v"] = next(it, last["v"])   # repeat final value if exhausted
        return last["v"]

    defaults = dict(threshold=0.855, min_consecutive=2, refractory_s=1.0)
    defaults.update(kwargs)
    return StreamingDetector(score_fn=score_fn, **defaults)


def feed_hops(det, n_hops):
    events = []
    for _ in range(n_hops):
        events.extend(det.process(np.zeros(HOP, np.float32)))
    return events


def test_no_fire_below_threshold():
    det = make_detector([0.5] * 20)
    assert feed_hops(det, 20) == []


def test_single_spike_suppressed_by_min_consecutive():
    """One hop above threshold must NOT fire (main FA source)."""
    det = make_detector([0.1, 0.99, 0.1, 0.1, 0.1])
    assert feed_hops(det, 5) == []


def test_two_consecutive_hops_fire_once():
    det = make_detector([0.1, 0.99, 0.99, 0.1, 0.1])
    events = feed_hops(det, 5)
    assert len(events) == 1
    assert events[0].score == pytest.approx(0.99)


def test_refractory_blocks_immediate_refire():
    """Sustained high score = one utterance = exactly one event within
    the refractory period."""
    det = make_detector([0.99] * 12, refractory_s=1.0)     # 12 hops = 1.2 s
    events = feed_hops(det, 12)
    assert len(events) <= 2      # one fire, refractory covers next 10 hops
    assert len(events) >= 1


def test_fires_again_after_refractory():
    scores = [0.99, 0.99] + [0.1] * 12 + [0.99, 0.99]
    det = make_detector(scores, refractory_s=1.0)
    events = feed_hops(det, len(scores))
    assert len(events) == 2
    assert events[1].time_s - events[0].time_s >= 1.0


def test_arbitrary_chunk_sizes_equivalent_to_hop_feeding():
    """Feeding 37-sample dribbles must produce identical inference count
    to feeding clean hops — the pending buffer must handle any chunking."""
    calls = {"n": 0}

    def counting_score(_x):
        calls["n"] += 1
        return 0.0

    det = StreamingDetector(score_fn=counting_score)
    total = HOP * 10
    fed = 0
    while fed < total:
        n = min(37, total - fed)
        det.process(np.zeros(n, np.float32))
        fed += n
    assert calls["n"] == 10


def test_buffer_is_true_sliding_window():
    """After feeding a marked sample, it must slide through and eventually
    leave the buffer after WINDOW samples."""
    det = StreamingDetector(score_fn=lambda _x: 0.0)
    marked = np.full(HOP, 7.0, np.float32)
    det.process(marked)
    assert (det._buf == 7.0).sum() == HOP
    det.process(np.zeros(WINDOW, np.float32))     # push a full window through
    assert (det._buf == 7.0).sum() == 0
