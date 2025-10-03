#!/usr/bin/env python3
"""
GPUWorker smoke test on CUDA using turbo model by default.
- Assumes cuDNN libs are on LD_LIBRARY_PATH (source scripts/gpu_env.sh).
- Feeds ~2s of audio and prints any emitted segments.
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_DIR)
from src.infra.gpu_worker import GPUWorker

AUDIO = os.path.join(REPO_DIR, 'audio.wav')
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

parser = argparse.ArgumentParser(description="GPUWorker CUDA smoke test")
parser.add_argument("--model", default="turbo", help="Whisper model id (default: turbo)")
parser.add_argument("--device-index", type=int, default=0, help="CUDA device index")
args = parser.parse_args()

print(f"Starting GPUWorker (CUDA, float16, model={args.model})...")
worker = GPUWorker(gpu_index=args.device_index, model_size=args.model, compute_type='float16', device='cuda',
                   max_sessions=1, language='en', use_vad=True, beam_size=1)
worker.create_session('call1')
worker.add_audio('call1', x[:32000])
outs = worker.process_next('call1')
print(f"Worker produced {len(outs)} outputs")
for o in outs[:5]:
    print(f" - [{o.start:.2f}..{o.end:.2f}] {o.text.strip()}")
