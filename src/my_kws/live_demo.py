"""Live wake-word detection from the microphone — the acceptance test.

Run from project root:
    uv run python -m my_kws.live_demo
    uv run python -m my_kws.live_demo --threshold 0.9   # stricter

Say "yes" at the mic. A score bar updates 10x per second; detections
print with a timestamp. Ctrl+C to stop; prints a session summary.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from .streaming import SR, Detection, StreamingDetector, make_onnx_score_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Live KWS demo")
    parser.add_argument("--model", default="models/dscnn_int8.onnx")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    import sounddevice as sd

    det = StreamingDetector(
        score_fn=make_onnx_score_fn(args.model),
        threshold=args.threshold,
        min_consecutive=args.min_consecutive,
    )

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    detections: list[Detection] = []
    t_start = time.time()
    print(f"listening (threshold {args.threshold}) — say \"yes\", Ctrl+C to stop")

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=SR // 10, device=args.device,
                        callback=callback):
        try:
            while True:
                chunk = q.get()
                events = det.process(chunk)
                bar = "#" * int(det.last_score * 30)
                print(f"\r score {det.last_score:5.3f} |{bar:<30s}|",
                      end="", flush=True)
                for e in events:
                    detections.append(e)
                    print(f"\n *** DETECTED @ {time.strftime('%H:%M:%S')} "
                          f"(score {e.score:.3f}) ***")
        except KeyboardInterrupt:
            pass

    dur = time.time() - t_start
    print(f"\n\nsession: {dur/60:.1f} min, {len(detections)} detections")


if __name__ == "__main__":
    main()
