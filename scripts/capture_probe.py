#!/usr/bin/env python3
"""
Quick capture probe: reads from a PyAudio input device and prints RMS/peak every interval.

Usage examples:
  python scripts/capture_probe.py --device 47
  python scripts/capture_probe.py --device-name pulse --suggest-on-fail
  PULSE_SOURCE=alsa_input.usb-...analog-stereo python scripts/capture_probe.py --device 47
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import pyaudio  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"PyAudio is required: {exc}")
    sys.exit(1)

RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = getattr(pyaudio, "paInt16", 8)  # type: ignore


def list_devices(pa: pyaudio.PyAudio) -> None:  # type: ignore[name-defined]
    print("Available audio devices (index | in_ch | out_ch | default_sample_rate | name):")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(
            f"{i:3d} | {info.get('maxInputChannels',0):3d} | {info.get('maxOutputChannels',0):3d} | {int(info.get('defaultSampleRate',0)):5d} | {info.get('name','Unknown')}"
        )


def find_device_index_by_name(pa: pyaudio.PyAudio, name: str, suggest: bool) -> int | None:  # type: ignore[name-defined]
    name_lower = name.lower()
    best_idx = None
    best_inputs = -1
    all_names: list[str] = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        nm = str(info.get("name", ""))
        all_names.append(nm)
        if name_lower in nm.lower() and info.get("maxInputChannels", 0) > 0:
            if info.get("maxInputChannels", 0) > best_inputs:
                best_idx = i
                best_inputs = info.get("maxInputChannels", 0)
    if best_idx is None and suggest:
        import difflib

        close = difflib.get_close_matches(name, all_names, n=5, cutoff=0.3)
        if close:
            print(f"❌ No match for '{name}'. Did you mean:")
            for c in close:
                print(f"   - {c}")
        else:
            print(f"❌ No input device matched substring '{name}'.")
    return best_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio capture probe: print RMS/peak")
    parser.add_argument("--device", type=int, help="Input device index")
    parser.add_argument("--device-name", help="Substring to choose input device by name")
    parser.add_argument("--suggest-on-fail", action="store_true", help="Suggest close device names on failure")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to run (default 10s)")
    parser.add_argument("--interval", type=float, default=0.5, help="Print interval seconds (default 0.5s)")
    parser.add_argument("--list", action="store_true", help="List devices and exit")
    args = parser.parse_args()

    pa = pyaudio.PyAudio()  # type: ignore[name-defined]
    if args.list:
        list_devices(pa)
        pa.terminate()
        return 0

    device_index = args.device
    if device_index is None and args.device_name:
        idx = find_device_index_by_name(pa, args.device_name, args.suggest_on_fail)
        if idx is None:
            pa.terminate()
            return 1
        device_index = idx

    print("Capture Probe")
    if dev := os.environ.get("PULSE_SOURCE"):
        print(f"PULSE_SOURCE={dev}")
    if device_index is not None:
        print(f"Input device index: {device_index}")

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    start = time.time()
    next_print = start
    count = 0
    try:
        while time.time() - start < args.seconds:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = math.sqrt(float(np.mean(audio ** 2))) if audio.size else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            count += 1
            now = time.time()
            if now >= next_print:
                print(f"t={now-start:5.2f}s | rms={rms:.4f} peak={peak:.4f}")
                next_print = now + args.interval
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
