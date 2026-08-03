"""Streaming keyword spotting — from classifier to wake-word system.

Engine:  StreamingDetector — feed arbitrary-sized chunks of 16 kHz audio,
         get back detection events. Testable without ONNX (inject score_fn).

Eval:    Synthetic-stream benchmark. Builds a long continuous stream from
         _background_noise_ with test-split "yes" clips injected at known
         times, then reports the two numbers that actually matter for a
         wake-word product:
             - recall on injected keywords
             - FA/hour (false accepts per hour) on everything else

Run from project root:
    uv run python -m my_kws.streaming --minutes 30

Design decisions (and their costs):
    window = 1.0 s      matches training input exactly
    hop    = 0.1 s      10 inferences/s; at 0.15 ms each => ~0.15% CPU budget
    features            full log-mel recompute per window. Wasteful (90% frame
                        overlap between consecutive windows) but simple and
                        still ~50x realtime; incremental STFT is a later
                        optimization chunk.
    trigger             score >= threshold for `min_consecutive` hops (default
                        2) — single-hop spikes are the main FA source.
    refractory          after firing, ignore `refractory_s` of audio so one
                        utterance produces one event, not ten.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import log_mel_spectrogram

SR = 16000
WINDOW = SR                      # 1.0 s
HOP = SR // 10                   # 0.1 s


@dataclass
class Detection:
    time_s: float                # stream time at which the trigger fired
    score: float


@dataclass
class StreamingDetector:
    """Feed chunks with process(); returns Detection events as they fire."""

    score_fn: callable                     # (1,1,40,98) float32 -> probability
    threshold: float = 0.98
    min_consecutive: int = 3
    refractory_s: float = 1.0

    _buf: np.ndarray = field(default_factory=lambda: np.zeros(WINDOW, np.float32))
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _samples_seen: int = 0
    _consecutive: int = 0
    _refractory_until: float = -1.0
    last_score: float = 0.0

    def process(self, chunk: np.ndarray) -> list[Detection]:
        """Consume any number of samples; run inference every HOP samples."""
        events: list[Detection] = []
        self._pending = np.concatenate(
            [self._pending, chunk.astype(np.float32, copy=False)]
        )
        while len(self._pending) >= HOP:
            hop, self._pending = self._pending[:HOP], self._pending[HOP:]
            self._buf = np.concatenate([self._buf[HOP:], hop])   # ring buffer
            self._samples_seen += HOP
            ev = self._step()
            if ev is not None:
                events.append(ev)
        return events

    def _step(self) -> Detection | None:
        now = self._samples_seen / SR
        feat = log_mel_spectrogram(self._buf).astype(np.float32)  # (40, 98)
        score = float(self.score_fn(feat[None, None]))
        self.last_score = score

        if now < self._refractory_until:
            self._consecutive = 0
            return None
        if score >= self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.min_consecutive:
                self._consecutive = 0
                self._refractory_until = now + self.refractory_s
                return Detection(time_s=now, score=score)
        else:
            self._consecutive = 0
        return None


def make_onnx_score_fn(onnx_path: str | Path):
    """Wrap an ONNX session (static batch=1) into score_fn."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])

    def score(x: np.ndarray) -> float:
        logit = sess.run(["logit"], {"log_mel": x})[0]
        return 1.0 / (1.0 + np.exp(-logit.item()))

    return score


# --------------------------------------------------------------------------
# Synthetic-stream benchmark
# --------------------------------------------------------------------------

def build_stream(minutes: float, data_root: Path, n_inject: int,
                 snr_db: float, seed: int = 0):
    """Long background stream + test-split 'yes' clips at known times.

    Returns (audio, inject_times_s). Background is tiled _background_noise_
    with random gain per segment; injections are mixed in (not spliced) so
    the keyword sits ON noise, like real usage.
    """
    import soundfile as sf

    rng = np.random.default_rng(seed)
    total = int(minutes * 60 * SR)

    noise_files = sorted((data_root / "_background_noise_").glob("*.wav"))
    stream = np.zeros(total, np.float32)
    pos = 0
    while pos < total:
        f = noise_files[rng.integers(len(noise_files))]
        seg, _ = sf.read(f, dtype="float32")
        if seg.ndim == 2:
            seg = seg.mean(axis=1)
        seg = seg * rng.uniform(0.3, 1.0)
        n = min(len(seg), total - pos)
        stream[pos:pos + n] = seg[:n]
        pos += n

    # pick test-split 'yes' files (never seen in training/val)
    test_list = {
        line.strip().replace("\\", "/")
        for line in (data_root / "testing_list.txt").read_text().splitlines()
        if line.strip().startswith("yes/")
    }
    yes_files = [data_root / rel for rel in sorted(test_list)]
    picks = rng.choice(len(yes_files), size=n_inject, replace=False)

    inject_times = np.sort(rng.uniform(2.0, minutes * 60 - 2.0, n_inject))
    # enforce >= 3 s spacing so events are unambiguous
    for i in range(1, n_inject):
        inject_times[i] = max(inject_times[i], inject_times[i - 1] + 3.0)

    for t, k in zip(inject_times, picks):
        clip, _ = sf.read(yes_files[int(k)], dtype="float32")
        if len(clip) < SR:
            clip = np.pad(clip, (0, SR - len(clip)))
        clip = clip[:SR]
        s = int(t * SR)
        bg_rms = float(np.sqrt(np.mean(stream[s:s + SR] ** 2)) + 1e-9)
        clip_rms = float(np.sqrt(np.mean(clip ** 2)) + 1e-9)
        gain = bg_rms / clip_rms * 10 ** (snr_db / 20)
        stream[s:s + SR] += clip * gain
    return stream, inject_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming KWS benchmark")
    parser.add_argument("--model", default="models/dscnn_int8.onnx")
    parser.add_argument("--data-root", default="data/raw/speech_commands")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--n-inject", type=int, default=60)
    parser.add_argument("--snr-db", type=float, default=10.0,
                        help="keyword level above background, in dB")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--match-window", type=float, default=1.5,
                        help="detection within this many seconds after an "
                             "injection counts as a hit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[stream] building {args.minutes:.0f} min stream, "
          f"{args.n_inject} injected keywords @ SNR {args.snr_db} dB ...")
    stream, inject_times = build_stream(
        args.minutes, Path(args.data_root), args.n_inject, args.snr_db,
        args.seed,
    )

    det = StreamingDetector(
        score_fn=make_onnx_score_fn(args.model),
        threshold=args.threshold,
        min_consecutive=args.min_consecutive,
    )

    t0 = time.perf_counter()
    events: list[Detection] = []
    for s in range(0, len(stream), SR):            # feed 1 s chunks
        events.extend(det.process(stream[s:s + SR]))
    wall = time.perf_counter() - t0
    audio_s = len(stream) / SR

    # match events to injections
    hits, used = 0, set()
    for t in inject_times:
        for j, e in enumerate(events):
            if j in used:
                continue
            if t <= e.time_s <= t + 1.0 + args.match_window:
                hits += 1
                used.add(j)
                break
    false_accepts = len(events) - len(used)
    fa_per_hour = false_accepts / (audio_s / 3600)

    print(f"\n[result] stream {audio_s/60:.1f} min | processed at "
          f"{audio_s/wall:.1f}x realtime")
    print(f"  injected keywords : {len(inject_times)}")
    print(f"  hits              : {hits}  (recall {hits/len(inject_times):.3f})")
    print(f"  false accepts     : {false_accepts}  "
          f"(FA/hour = {fa_per_hour:.2f})")
    print(f"  total detections  : {len(events)}")


if __name__ == "__main__":
    main()
