"""
Transcribe a segment from an audio file using Whimper
"""

import sys
import os
import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T
from torchaudio import functional as AF
import argparse
import json


def _format_srt_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_segments_json(segments, path: str, pretty: bool = True) -> None:
    payload = [
        {
            "start": float(getattr(seg, "start", 0.0)),
            "end": float(getattr(seg, "end", 0.0)),
            "text": getattr(seg, "text", str(seg)),
            "is_final": bool(getattr(seg, "is_final", True)),
        }
        for seg in segments
    ]
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False)


def _write_segments_srt(segments, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", 0.0))
            text = getattr(seg, "text", str(seg))
            f.write(f"{i}\n")
            f.write(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n")
            f.write(text.strip() + "\n\n")

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whimper import GPULiveTranscriber, TranscriptionResult

def transcription_callback(result: TranscriptionResult | str) -> None:
    if isinstance(result, TranscriptionResult):
        status = "FINAL" if result.is_final else "LIVE"
        print(f"[{result.start:.2f}s - {result.end:.2f}s] {status}: {result.text}")
    else:
        print(f"FINAL: {result}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe an audio file segment with Whimper (torchaudio pipeline)")
    parser.add_argument("--audio-file", "-i", default="audio.wav", help="Path to input audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: to end)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (e.g., tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="Inference device")
    parser.add_argument("--language", default="en", help="Language code (e.g., en, es, auto if supported)")
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument("--vad", dest="vad", action="store_true", help="Enable VAD trimming before ASR")
    vad_group.add_argument("--no-vad", dest="vad", action="store_false", help="Disable VAD trimming before ASR")
    parser.set_defaults(vad=True)
    parser.add_argument("--out-json", default=None, help="Optional path to write segments as JSON")
    parser.add_argument("--out-srt", default=None, help="Optional path to write segments as SRT subtitles")
    parser.add_argument("--pretty-json", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_file = args.audio_file
    segment_start = args.start
    segment_duration = args.duration
    target_sr = args.sr

    print(f"Loading audio file: {audio_file}")
    # Load audio with torchaudio (keeping everything on CPU)
    wav, sr = torchaudio.load(audio_file)  # wav: [channels, time]

    # Convert to mono if needed
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Optional segment slicing (in input SR), then resample
    if segment_start or (segment_duration is not None):
        start_sample = int(segment_start * sr)
        end_sample = int((segment_start + (segment_duration or (wav.size(1) / sr))) * sr)
        end_sample = min(end_sample, wav.size(1))
        wav = wav[:, start_sample:end_sample]

    # Resample to 16kHz mono
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    # Apply VAD trimming using torchaudio VAD (preferred)
    # If unavailable or it fails, leave audio as-is
    if args.vad:
        try:
            # torchaudio.functional.vad expects mono. Shape can be [time] or [1, time].
            if wav.dim() == 2 and wav.size(0) == 1:
                wav = AF.vad(wav, sample_rate=sr)
            else:
                wav = AF.vad(wav.squeeze(0), sample_rate=sr).unsqueeze(0)
            # If VAD trimmed everything (edge case), skip trimming
            if wav.numel() == 0:
                raise RuntimeError("VAD produced empty output")
        except Exception:
            pass

    # Ensure shape [time] and type float32 numpy in -1..1
    wav = wav.squeeze(0).contiguous()
    audio = wav.detach().cpu().numpy().astype(np.float32)

    print(f"Audio loaded: {len(audio)} samples at {sr} Hz (torchaudio{' + VAD' if args.vad else ''})")
    
    # Create transcriber
    # Initialize transcriber with CLI options
    init_kwargs = dict(
        model_size=args.model,
        language=args.language,
        device=args.device,
        callback=transcription_callback,
    )
    try:
        transcriber = GPULiveTranscriber(**init_kwargs)
    except TypeError:
        # Fallback without device if the signature differs
        init_kwargs.pop("device", None)
        transcriber = GPULiveTranscriber(**init_kwargs)
    
    # Get the session
    session = transcriber.session

    # Collect segments as we go for optional export
    collected = []

    def collecting_callback(result):
        # Reuse printing behavior and also collect
        transcription_callback(result)
        # We only collect segment-like results (not plain strings) for export
        collected.append(result)

    # Swap callback temporarily if supported
    try:
        session.callback = collecting_callback  # type: ignore[attr-defined]
    except Exception:
        pass

    # Add the entire audio to the session
    session.add_audio(audio.astype(np.float32))

    # Process until no more
    while True:
        results = session.process_next()
        if not results:
            break
        for result in results:
            collecting_callback(result)

    # Finalize if needed
    results = session.process_next()
    for result in results:
        collecting_callback(result)

    # Write outputs if requested
    if args.out_json:
        _write_segments_json(collected, args.out_json, pretty=args.pretty_json)
        print(f"Wrote JSON segments to {args.out_json}")
    if args.out_srt:
        _write_segments_srt(collected, args.out_srt)
        print(f"Wrote SRT to {args.out_srt}")
    
    # No close

if __name__ == "__main__":
    main()
