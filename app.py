"""Gradio web demo for my_KWS — browser-mic streaming wake-word detection.

Architecture (compare with live_demo.py):

    live_demo.py : sounddevice mic -> queue -> StreamingDetector -> terminal
    app.py       : browser mic -> Gradio stream events -> StreamingDetector -> HTML UI

StreamingDetector is reused UNCHANGED. The only genuinely new problems a web
front-end introduces are handled here, each marked with [WEB-n] comments:

    [WEB-1] sample-rate mismatch: browsers record at 48 kHz (sometimes 44.1),
            int16, possibly stereo. The whole KWS chain assumes 16 kHz mono
            float32 in [-1, 1], so every chunk is converted on arrival.
    [WEB-2] per-session state: the detector holds a ring buffer + refractory
            timer, so each visitor needs their OWN instance (gr.State).
            A global detector would mix audio from concurrent users.
    [WEB-3] event-driven streaming: Gradio calls `on_chunk` once per audio
            chunk. This is live_demo.py's `while True: q.get()` loop turned
            inside-out — state goes in, state + UI come back out.

Run locally:   python app.py           (needs models/dscnn_int8.onnx)
Deploy:        push this folder to a Hugging Face Space (SDK: gradio)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent / "src"))

from my_kws.streaming import SR, StreamingDetector, make_onnx_score_fn  # noqa: E402

import gradio as gr  # noqa: E402

MODEL_PATH = Path(__file__).parent / "models" / "dscnn_int8.onnx"
KEYWORD = "yes"
HISTORY_LEN = 100          # score points kept for the sparkline (~10 s)


# --------------------------------------------------------------------------
# [WEB-1] browser chunk -> 16 kHz mono float32
# --------------------------------------------------------------------------

def to_16k_mono_f32(sr: int, y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:                       # stereo -> mono
        y = y.mean(axis=1)
    if y.dtype == np.int16:               # int16 -> float32 in [-1, 1]
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32, copy=False)
    if sr != SR:
        g = np.gcd(sr, SR)                # 48000->16000 is a clean /3
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


# --------------------------------------------------------------------------
# [WEB-2] one detector (+ UI history) per session
# --------------------------------------------------------------------------

def new_session(threshold: float = 0.98) -> dict:
    return {
        "det": StreamingDetector(
            score_fn=make_onnx_score_fn(MODEL_PATH),
            threshold=threshold,
            min_consecutive=3,
        ),
        "scores": [],        # rolling score history for the sparkline
        "events": [],        # (stream_time_s, score) of past detections
    }


# --------------------------------------------------------------------------
# UI rendering (pure string building — no extra plotting deps)
# --------------------------------------------------------------------------

def render_panel(state: dict) -> str:
    det: StreamingDetector = state["det"]
    score = det.last_score
    scores = state["scores"]
    thr = det.threshold

    hot = score >= thr
    bar_color = "#e74c3c" if hot else "#2ecc71"
    bar_pct = int(score * 100)

    # sparkline as inline SVG
    pts = scores[-HISTORY_LEN:]
    if len(pts) >= 2:
        w, h = 560, 80
        xs = np.linspace(0, w, len(pts))
        ys = h - np.asarray(pts) * h
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        thr_y = h - thr * h
        spark = (
            f'<svg width="{w}" height="{h}" style="background:#111;border-radius:6px">'
            f'<line x1="0" y1="{thr_y:.1f}" x2="{w}" y2="{thr_y:.1f}" '
            f'stroke="#e74c3c" stroke-dasharray="4,4" stroke-width="1"/>'
            f'<polyline points="{poly}" fill="none" stroke="#2ecc71" stroke-width="2"/>'
            f"</svg>"
        )
    else:
        spark = '<div style="color:#888">waiting for audio…</div>'

    rows = "".join(
        f"<li><b>DETECTED</b> @ {t:6.1f}s &nbsp; score {s:.3f}</li>"
        for t, s in reversed(state["events"][-8:])
    ) or "<li style='color:#888'>none yet — say the keyword</li>"

    return f"""
    <div style="font-family:monospace">
      <div style="margin-bottom:6px">score {score:5.3f} / threshold {thr:.2f}</div>
      <div style="background:#333;border-radius:6px;height:22px;width:560px">
        <div style="background:{bar_color};height:22px;border-radius:6px;
                    width:{bar_pct}%;transition:width .1s"></div>
      </div>
      <div style="margin:10px 0">{spark}</div>
      <ul style="padding-left:18px">{rows}</ul>
    </div>
    """


# --------------------------------------------------------------------------
# [WEB-3] stream callback: (state, chunk) -> (state, html)
# --------------------------------------------------------------------------

def on_chunk(state: dict | None, chunk: tuple[int, np.ndarray] | None,
             threshold: float):
    if state is None:
        state = new_session(threshold)
    if chunk is None:
        return state, render_panel(state)

    state["det"].threshold = float(threshold)   # live-tunable
    sr, y = chunk
    audio = to_16k_mono_f32(sr, y)

    events = state["det"].process(audio)        # <- the entire KWS system
    state["scores"].append(state["det"].last_score)
    state["scores"] = state["scores"][-HISTORY_LEN:]
    state["events"].extend((e.time_s, e.score) for e in events)

    return state, render_panel(state)


def reset(threshold: float):
    s = new_session(threshold)
    return s, render_panel(s)


with gr.Blocks(title="my_KWS — streaming wake-word demo") as demo:
    gr.Markdown(
        f"# my_KWS — streaming keyword spotting\n"
        f"Say **\u201c{KEYWORD}\u201d** into the mic. int8-quantized DS-CNN "
        f"(~65k params) running on CPU via onnxruntime; 1 s window, 100 ms hop, "
        f"{3}-hop debounce, 1 s refractory."
    )
    session = gr.State(None)
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                       label="microphone")
        with gr.Column():
            thr = gr.Slider(0.5, 0.999, value=0.98, step=0.001,
                            label="threshold (higher = fewer false accepts)")
            clear = gr.Button("reset session")
    panel = gr.HTML(render_panel(new_session()))

    mic.stream(on_chunk, inputs=[session, mic, thr], outputs=[session, panel],
               stream_every=0.2, time_limit=120)
    clear.click(reset, inputs=[thr], outputs=[session, panel])

if __name__ == "__main__":
    demo.launch()
