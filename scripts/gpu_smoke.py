#!/usr/bin/env python3
"""
GPU smoke test for faster-whisper + cuDNN via CUDA.
- Assumes you have sourced scripts/gpu_env.sh to expose cuDNN libs.
- Defaults to the 'turbo' model; override with --model.
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AUDIO = os.path.join(REPO_DIR, 'audio.wav')

parser = argparse.ArgumentParser(description="CUDA smoke test for faster-whisper")
parser.add_argument("--model", default="turbo", help="Whisper model id (default: turbo)")
parser.add_argument("--device-index", type=int, default=0, help="CUDA device index")
args = parser.parse_args()

if not os.path.exists(AUDIO):
    print(f"ERROR: Missing audio file: {AUDIO}")
    sys.exit(1)

x, sr = sf.read(AUDIO, dtype='float32', always_2d=False)
if x.ndim > 1:
    x = x.mean(axis=1)
if sr != 16000:
    t = np.arange(len(x)) / sr
    new_t = np.linspace(0, t[-1], int(len(x) * 16000 / sr))
    x = np.interp(new_t, t, x).astype('float32')

print(f"Loading {args.model} on CUDA (float16)...")
model = WhisperModel(args.model, device='cuda', compute_type='float16', device_index=args.device_index)
segs, info = model.transcribe(x[:16000], beam_size=1, vad_filter=False)
segs = list(segs)
print(f"OK: {len(segs)} segments")
for s in segs[:3]:
    print(f" - [{getattr(s, 'start', 0.0):.2f}..{getattr(s, 'end', 0.0):.2f}] {getattr(s, 'text','').strip()}")
