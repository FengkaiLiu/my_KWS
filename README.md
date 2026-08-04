---
title: my_KWS streaming demo
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
---

# my_KWS — live streaming keyword spotting

Browser-mic demo of a from-scratch KWS system:
hand-written log-mel front-end (numpy) → DS-CNN (~65k params) →
int8 ONNX → streaming detector (1 s window / 100 ms hop, debounce + refractory).

Source: https://github.com/FengkaiLiu/my_KWS
