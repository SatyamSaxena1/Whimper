"""
Whimper - GPU-Accelerated Live Audio Transcription with Whisper v3

This module provides a high-performance real-time speech-to-text engine inspired by
Collabora's WhisperLive project. Audio is captured from the microphone, buffered in a
streaming session, and transcribed incrementally using a GPU-accelerated
faster-whisper backend.
"""

from __future__ import annotations

import logging
import os
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

# Optional: torch is only needed for GPU device detection in GPULiveTranscriber
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None  # type: ignore

# Core deps that must exist
import numpy as np
from faster_whisper import WhisperModel

# Optional: PyAudio is only required for live microphone capture
try:  # pragma: no cover - optional dependency
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover - allow import without pyaudio
    pyaudio = None  # type: ignore

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AudioConfig:
    """Audio configuration constants optimised for Whisper."""

    SAMPLE_RATE = 16000  # Whisper expects 16 kHz audio
    CHANNELS = 1  # mono audio
    CHUNK_SIZE = 1024  # frames captured per callback
    # 16-bit PCM samples; if PyAudio is unavailable, use a safe placeholder
    FORMAT = getattr(pyaudio, "paInt16", 8)  # type: ignore
    BYTES_PER_SAMPLE = 2


@dataclass
class TranscriptionResult:
    """Structured representation of a transcription event."""

    text: str
    start: float
    end: float
    is_final: bool
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    def pretty_status(self) -> str:
        return "FINAL" if self.is_final else "LIVE"


class SimpleVAD:
    """Simple energy based voice activity detector used as a lightweight fallback."""

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        energy_threshold: float = 0.010,
        min_active_frames: int = 3,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.energy_threshold = energy_threshold
        self.min_active_frames = max(1, min_active_frames)

    def contains_voice(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False

        active_frames = 0
        for index in range(0, audio.shape[0], self.frame_samples):
            frame = audio[index : index + self.frame_samples]
            if frame.size == 0:
                break
            rms = math.sqrt(float(np.mean(frame**2))) if frame.size else 0.0
            if rms > self.energy_threshold:
                active_frames += 1
                if active_frames >= self.min_active_frames:
                    return True
        return False


class StreamingTranscriptionSession:
    """Real-time transcription session inspired by WhisperLive's streaming server."""

    RATE = AudioConfig.SAMPLE_RATE
    MAX_BUFFER_SECONDS = 45.0
    DROP_SECONDS = 30.0
    MIN_CHUNK_SECONDS = 0.6
    MAX_CHUNK_SECONDS = 6.0
    SILENCE_STEP_SECONDS = 0.5
    SAME_OUTPUT_THRESHOLD = 4

    def __init__(
        self,
        model: WhisperModel,
        language: str = "en",
        use_vad: bool = True,
        no_speech_threshold: float = 0.45,
        same_output_threshold: Optional[int] = None,
        beam_size: int = 5,
        min_chunk_seconds: Optional[float] = None,
        max_chunk_seconds: Optional[float] = None,
        disable_internal_vad: bool = False,
        debug_audio_levels: bool = False,
    ) -> None:
        self.model = model
        self.language_setting = language
        self.detected_language: Optional[str] = None
        self.use_vad = use_vad
        self.no_speech_threshold = no_speech_threshold
        self.same_output_threshold = (
            same_output_threshold
            if same_output_threshold is not None
            else self.SAME_OUTPUT_THRESHOLD
        )
        self.beam_size = beam_size
        # Per-session chunk sizing controls
        self.min_chunk_seconds = float(min_chunk_seconds) if min_chunk_seconds is not None else self.MIN_CHUNK_SECONDS
        self.max_chunk_seconds = float(max_chunk_seconds) if max_chunk_seconds is not None else self.MAX_CHUNK_SECONDS
        if self.min_chunk_seconds <= 0:
            self.min_chunk_seconds = self.MIN_CHUNK_SECONDS
        if self.max_chunk_seconds <= 0:
            self.max_chunk_seconds = self.MAX_CHUNK_SECONDS
        if self.min_chunk_seconds > self.max_chunk_seconds:
            # swap to keep sane ordering
            self.min_chunk_seconds, self.max_chunk_seconds = self.max_chunk_seconds, self.min_chunk_seconds

        # VAD behaviors
        self.disable_internal_vad = bool(disable_internal_vad)
        self.debug_audio_levels = bool(debug_audio_levels)

        self.lock = threading.Lock()
        self.frames_np: Optional[np.ndarray] = None
        self.frames_offset = 0.0
        self.timestamp_offset = 0.0

        self.transcript: List[TranscriptionResult] = []
        self._sent_final_keys: set[Tuple[float, float, str]] = set()
        self._last_partial_key: Optional[Tuple[float, str]] = None
        self._previous_partial_text: str = ""
        self._partial_repeat_count = 0
        self._last_processed_samples: Optional[int] = None

        self.vad = SimpleVAD(self.RATE) if self.use_vad else None

    # ------------------------------------------------------------------
    # Buffer management helpers
    # ------------------------------------------------------------------
    def add_audio(self, audio_frame: np.ndarray) -> None:
        """Append new audio samples (float32 in range [-1, 1]) to the buffer."""
        if audio_frame.size == 0:
            return

        if audio_frame.ndim != 1:
            audio_frame = audio_frame.reshape(-1)

        with self.lock:
            if self.frames_np is None:
                self.frames_np = audio_frame.copy()
            else:
                self.frames_np = np.concatenate((self.frames_np, audio_frame))

            max_samples = int(self.MAX_BUFFER_SECONDS * self.RATE)
            if self.frames_np.shape[0] > max_samples:
                drop_samples = int(self.DROP_SECONDS * self.RATE)
                self.frames_np = self.frames_np[drop_samples:]
                self.frames_offset += self.DROP_SECONDS
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset
            self._last_processed_samples = None

    def _get_audio_chunk(self) -> Tuple[np.ndarray, float]:
        with self.lock:
            if self.frames_np is None:
                return np.array([], dtype=np.float32), 0.0

            consumed_seconds = max(0.0, self.timestamp_offset - self.frames_offset)
            start_index = int(consumed_seconds * self.RATE)
            if start_index >= self.frames_np.shape[0]:
                return np.array([], dtype=np.float32), 0.0

            chunk = self.frames_np[start_index:].copy()

        max_samples = int(self.max_chunk_seconds * self.RATE)
        if chunk.shape[0] > max_samples:
            chunk = chunk[:max_samples]

        return chunk, chunk.shape[0] / self.RATE

    # ------------------------------------------------------------------
    # Streaming transcription logic
    # ------------------------------------------------------------------
    def process_next(self) -> List[TranscriptionResult]:
        chunk, duration = self._get_audio_chunk()
        samples = int(chunk.shape[0])

        if samples == 0:
            self._last_processed_samples = None
            return []

        if duration < self.min_chunk_seconds:
            return []

        if (
            self._last_processed_samples is not None
            and samples == self._last_processed_samples
        ):
            return []

        # Optional audio diagnostics
        if self.debug_audio_levels:
            # Compute simple RMS and peak in [-1, 1]
            rms = math.sqrt(float(np.mean(chunk**2))) if chunk.size else 0.0
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            voice_decision = None
            if self.vad is not None:
                voice_decision = self.vad.contains_voice(chunk)
            else:
                # Heuristic decision when no external VAD: compare to SimpleVAD threshold
                voice_decision = rms > SimpleVAD(self.RATE).energy_threshold
            logger.info("[AUDIO] dur=%.2fs rms=%.4f peak=%.4f voice=%s", duration, rms, peak, str(bool(voice_decision)))

        if self.use_vad and self.vad and not self.vad.contains_voice(chunk):
            self.timestamp_offset += min(duration, self.SILENCE_STEP_SECONDS)
            self._last_processed_samples = None
            return []

        segments, info = self._transcribe_chunk(chunk)
        if info is not None and self.detected_language is None:
            language = getattr(info, "language", None)
            probability = getattr(info, "language_probability", 0.0)
            if language and probability >= 0.5:
                self.detected_language = language
                logger.info("🔤 Detected language: %s (p=%.2f)", language, probability)

        if not segments:
            self.timestamp_offset += min(duration, self.SILENCE_STEP_SECONDS)
            self._last_processed_samples = None
            return []

        results, advance = self._handle_segments(segments, duration)
        if advance > 0:
            self.timestamp_offset += advance
            self._last_processed_samples = None
        else:
            self._last_processed_samples = samples
        return results

    # ------------------------------------------------------------------
    # Internal helpers inspired by WhisperLive's ServeClientBase
    # ------------------------------------------------------------------
    def _transcribe_chunk(self, chunk: np.ndarray) -> Tuple[List[object], Optional[object]]:
        language = self.detected_language or None
        if self.language_setting not in (None, "auto"):
            language = self.language_setting

        output = self.model.transcribe(
            chunk,
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=False,
            # If disable_internal_vad is True, force internal VAD off.
            vad_filter=False if self.disable_internal_vad else (not self.use_vad),
        )

        if isinstance(output, tuple):
            segments, info = output
        else:  # pragma: no cover - legacy API fallback
            segments, info = output, None

        if isinstance(segments, Iterable) and not isinstance(segments, list):
            segments = list(segments)
        elif segments is None:
            segments = []

        return segments, info

    @staticmethod
    def _segment_start(segment: object) -> float:
        return float(getattr(segment, "start", getattr(segment, "start_ts", 0.0)))

    @staticmethod
    def _segment_end(segment: object) -> float:
        return float(getattr(segment, "end", getattr(segment, "end_ts", 0.0)))

    @staticmethod
    def _segment_no_speech(segment: object) -> float:
        return float(getattr(segment, "no_speech_prob", 0.0))

    def _segment_has_speech(self, segment: object) -> bool:
        return self._segment_no_speech(segment) <= self.no_speech_threshold

    @staticmethod
    def _final_key(start: float, end: float, text: str) -> Tuple[float, float, str]:
        return (round(start, 2), round(end, 2), text.strip())

    @staticmethod
    def _partial_key(start: float, text: str) -> Tuple[float, str]:
        return (round(start, 2), text.strip())

    def _update_partial_repeat(self, text: str) -> None:
        normalised = text.strip().lower()
        if not normalised:
            self._reset_partial_repeat()
            return
        if normalised == self._previous_partial_text:
            self._partial_repeat_count += 1
        else:
            self._previous_partial_text = normalised
            self._partial_repeat_count = 1

    def _reset_partial_repeat(self) -> None:
        self._previous_partial_text = ""
        self._partial_repeat_count = 0

    def _finalize_partial_if_needed(
        self, partial: Optional[TranscriptionResult]
    ) -> Optional[TranscriptionResult]:
        if partial is None:
            return None
        if self._partial_repeat_count < self.same_output_threshold:
            return None

        key = self._final_key(partial.start, partial.end, partial.text)
        if key in self._sent_final_keys:
            return None

        final_result = TranscriptionResult(
            text=partial.text,
            start=partial.start,
            end=partial.end,
            is_final=True,
            confidence=partial.confidence,
        )
        self._sent_final_keys.add(key)
        self.transcript.append(final_result)
        self._last_partial_key = None
        self._reset_partial_repeat()
        return final_result

    def _handle_segments(
        self, segments: Sequence[object], duration: float
    ) -> Tuple[List[TranscriptionResult], float]:
        results: List[TranscriptionResult] = []
        advance = 0.0

        for segment in segments[:-1]:
            if not self._segment_has_speech(segment):
                continue

            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue

            start = self.timestamp_offset + self._segment_start(segment)
            end = self.timestamp_offset + min(duration, self._segment_end(segment))
            if end <= start:
                continue

            key = self._final_key(start, end, text)
            if key in self._sent_final_keys:
                continue

            confidence = max(0.0, 1.0 - self._segment_no_speech(segment))
            result = TranscriptionResult(text=text, start=start, end=end, is_final=True, confidence=confidence)
            self._sent_final_keys.add(key)
            self.transcript.append(result)
            results.append(result)
            advance = max(advance, min(duration, self._segment_end(segment)))

        partial: Optional[TranscriptionResult] = None
        last_segment = segments[-1]
        if self._segment_has_speech(last_segment):
            text = str(getattr(last_segment, "text", "")).strip()
            if text:
                start = self.timestamp_offset + self._segment_start(last_segment)
                end = self.timestamp_offset + min(duration, self._segment_end(last_segment))
                if end > start:
                    confidence = max(0.0, 1.0 - self._segment_no_speech(last_segment))
                    partial = TranscriptionResult(
                        text=text,
                        start=start,
                        end=end,
                        is_final=False,
                        confidence=confidence,
                    )
                    self._update_partial_repeat(text)
                    key = self._partial_key(start, text)
                    if key != self._last_partial_key:
                        results.append(partial)
                        self._last_partial_key = key
            else:
                self._reset_partial_repeat()
                self._last_partial_key = None
        else:
            self._reset_partial_repeat()
            self._last_partial_key = None

        finalized = self._finalize_partial_if_needed(partial)
        if finalized:
            results.append(finalized)
            advance = max(advance, finalized.end - self.timestamp_offset)

        return results, advance


class GPULiveTranscriber:
    """High performance GPU accelerated live audio transcription."""

    def __init__(
        self,
        model_size: str = "large-v3",
        model_path: Optional[str] = None,
        language: str = "en",
        device: str = "auto",
        compute_type: str = "auto",
        callback: Optional[Callable[[TranscriptionResult], None]] = None,
        use_vad: bool = True,
        device_index: Optional[int] = None,
        # Session tuning
        min_chunk_seconds: Optional[float] = None,
        max_chunk_seconds: Optional[float] = None,
        disable_internal_vad: bool = False,
        debug_audio_levels: bool = False,
    ) -> None:
        self.model_size = model_size
        self.model_path = model_path
        self.language = language
        self.callback = callback
        self.use_vad = use_vad
        self.device_index = device_index
        self.min_chunk_seconds = min_chunk_seconds
        self.max_chunk_seconds = max_chunk_seconds
        self.disable_internal_vad = disable_internal_vad
        self.debug_audio_levels = debug_audio_levels

        self.device, self.compute_type = self._determine_device_config(device, compute_type)

        # Use Any to avoid type issues when PyAudio is not installed
        self.audio: Optional[Any] = None
        self.stream: Optional[Any] = None
        self.model: Optional[WhisperModel] = None
        self.session: Optional[StreamingTranscriptionSession] = None

        self.is_running = False
        self.is_recording = False
        self.processing_thread: Optional[threading.Thread] = None

        self.transcription_count = 0
        self.total_processing_time = 0.0
        self.final_segments_count = 0

        self._initialize_audio()
        self._initialize_whisper()
        self._initialize_session()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _determine_device_config(self, device: str, compute_type: str) -> Tuple[str, str]:
        cuda_available = False
        try:
            if torch is not None and getattr(torch, "cuda", None):
                cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        if device == "auto":
            actual_device = "cuda" if cuda_available else "cpu"
            if actual_device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name() if torch else "CUDA"
                except Exception:
                    gpu_name = "CUDA"
                logger.info("🚀 CUDA detected! GPU: %s", gpu_name)
            else:
                logger.info("💻 Using CPU (CUDA not available)")
        else:
            actual_device = device

        if compute_type == "auto":
            if actual_device == "cuda":
                actual_compute_type = "float16"
            else:
                actual_compute_type = "int8"
        else:
            actual_compute_type = compute_type

        logger.info("⚙️ Device: %s, Compute type: %s", actual_device, actual_compute_type)
        return actual_device, actual_compute_type

    def _initialize_audio(self) -> None:
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("🎤 Available audio input devices:")
            device_count = 0
            for index in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(index)
                if info.get("maxInputChannels", 0) > 0:
                    device_count += 1
                    if device_count <= 5:
                        logger.info("   %s: %s", index, info.get("name", "Unknown"))
            if device_count == 0:
                logger.warning("No audio input devices detected by PyAudio")
        except Exception as exc:  # pragma: no cover - hardware interaction
            logger.error("❌ Failed to initialise audio: %s", exc)
            raise

    def _initialize_whisper(self) -> None:
        if self.model_path:
            model_source = self.model_path
            logger.info("🤖 Loading local Whisper model from %s", model_source)
            local_files_only = True
        else:
            model_source = self.model_size
            local_files_only = False
            logger.info("🤖 Loading Whisper model '%s' (GPU accelerated)", model_source)

        start = time.time()
        extra: dict[str, object] = {}
        if self.device == "cuda" and self.device_index is not None:
            extra["device_index"] = self.device_index
        self.model = WhisperModel(
            model_source,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=local_files_only,
            **extra,
        )
        elapsed = time.time() - start
        logger.info("✅ Whisper model ready in %.1fs", elapsed)

    def _initialize_session(self) -> None:
        if self.model is None:
            raise RuntimeError("Whisper model not initialised")
        self.session = StreamingTranscriptionSession(
            self.model,
            language=self.language,
            use_vad=self.use_vad,
            min_chunk_seconds=self.min_chunk_seconds,
            max_chunk_seconds=self.max_chunk_seconds,
            disable_internal_vad=self.disable_internal_vad,
            debug_audio_levels=self.debug_audio_levels,
        )

    # ------------------------------------------------------------------
    # Audio callback and processing thread
    # ------------------------------------------------------------------
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            logger.warning("⚠️ Audio callback status: %s", status)
        if not self.session:
            return in_data, pyaudio.paContinue

        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0
        self.session.add_audio(audio_data)
        return in_data, pyaudio.paContinue

    def _processing_thread(self) -> None:
        logger.info("🔄 Streaming transcription thread started")
        while self.is_running:
            try:
                if not self.session:
                    time.sleep(0.1)
                    continue

                start_time = time.time()
                results = self.session.process_next()
                duration = time.time() - start_time

                if results:
                    for result in results:
                        self._handle_transcription_result(result)
                    self.transcription_count += 1
                    self.total_processing_time += duration
                else:
                    time.sleep(0.05)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("❌ Error in processing thread: %s", exc)
                time.sleep(0.1)

    def _handle_transcription_result(self, result: TranscriptionResult) -> None:
        if not result.text:
            return

        if self.callback:
            try:
                self.callback(result)
            except TypeError:
                prefix = "[FINAL]" if result.is_final else "[LIVE]"
                self.callback(f"{prefix} {result.text}")
        else:
            timestamp = time.strftime("%H:%M:%S")
            prefix = "✅" if result.is_final else "📝"
            print(f"{prefix} [{timestamp}] {result.text}")
        # Track final segments for no-output warning logic
        try:
            if isinstance(result, TranscriptionResult) and result.is_final:
                self.final_segments_count += 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public control methods
    # ------------------------------------------------------------------
    def start_recording(self, device_index: Optional[int] = None) -> None:
        if self.is_recording:
            logger.warning("⚠️ Recording already in progress")
            return

        if not self.audio:
            raise RuntimeError("PyAudio not initialised")

        try:
            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=AudioConfig.CHUNK_SIZE,
                stream_callback=self._audio_callback,
            )
            self.is_running = True
            self.is_recording = True

            self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
            self.processing_thread.start()
            self.stream.start_stream()
            logger.info("🎤 Live transcription started")
        except Exception as exc:
            logger.error("❌ Failed to start recording: %s", exc)
            self.stop_recording()
            raise

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        logger.info("🛑 Stopping live transcription")
        self.is_running = False
        self.is_recording = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as exc:  # pragma: no cover - hardware interaction
                logger.warning("⚠️ Error stopping audio stream: %s", exc)
            finally:
                self.stream = None

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None

        if self.transcription_count:
            avg = self.total_processing_time / max(1, self.transcription_count)
            logger.info(
                "📊 Processing stats: %s iterations, avg %.2fs",
                self.transcription_count,
                avg,
            )

    def cleanup(self) -> None:
        self.stop_recording()
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as exc:  # pragma: no cover - hardware interaction
                logger.warning("⚠️ Error terminating PyAudio: %s", exc)
            finally:
                self.audio = None

    def __enter__(self) -> "GPULiveTranscriber":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def gpu_transcription_callback(result: TranscriptionResult | str) -> None:
    """Default callback used by the CLI examples."""

    timestamp = time.strftime("%H:%M:%S")
    if isinstance(result, TranscriptionResult):
        status = result.pretty_status()
        text = result.text
    else:
        status = "FINAL"
        text = str(result).strip()
    print(f"🚀🎤 [{timestamp}] {status} {text}")


def _format_srt_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_segments_json(segments: List[TranscriptionResult | str], path: str, pretty: bool = True) -> None:
    import json

    payload = []
    for seg in segments:
        if isinstance(seg, TranscriptionResult):
            payload.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text,
                    "is_final": bool(seg.is_final),
                    "confidence": float(getattr(seg, "confidence", 1.0)),
                }
            )
        else:
            payload.append({"text": str(seg)})
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False)


def _write_segments_srt(segments: List[TranscriptionResult | str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        index = 1
        for seg in segments:
            if not isinstance(seg, TranscriptionResult):
                continue
            start = float(seg.start)
            end = float(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{index}\n")
            f.write(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n")
            f.write(text + "\n\n")
            index += 1


def main() -> int:
    """Command line entry point for live transcription."""

    import argparse

    parser = argparse.ArgumentParser(description="Whimper - Live GPU transcription")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["large-v3", "large-v2", "large", "medium", "small", "base", "tiny", "turbo"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument("--model-path", help="Path to a local Whisper model directory")
    parser.add_argument("--language", default="en", help="Source language code or 'auto'")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8", "float32"],
        help="Compute precision",
    )
    parser.add_argument("--audio-device", type=int, help="PyAudio device index to use")
    parser.add_argument("--audio-device-name", help="Substring match to select input device by name (case-insensitive)")
    parser.add_argument("--list-audio-devices", action="store_true", help="List audio input/output devices and exit")
    parser.add_argument("--output-device", type=int, help="(Future) Output/playback device index (unused now)")
    parser.add_argument("--suggest-on-fail", action="store_true", help="When name not found, print close suggestions instead of exiting immediately")
    parser.add_argument("--pulse-source", help="Explicit PulseAudio source name (sets PULSE_SOURCE env) e.g. alsa_input.usb-...analog-stereo")
    parser.add_argument("--no-vad", action="store_true", help="Disable energy based VAD filtering")
    parser.add_argument("--device-index", type=int, help="CUDA device index when using --device cuda")
    parser.add_argument("--out-json", default=None, help="Optional path to write final segments as JSON when exiting")
    parser.add_argument("--out-srt", default=None, help="Optional path to write final segments as SRT when exiting")
    parser.add_argument("--pretty-json", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (lower = faster, default 5)")
    parser.add_argument("--min-chunk-seconds", type=float, default=None, help="Minimum seconds of audio before transcribing a chunk (default 0.6)")
    parser.add_argument("--max-chunk-seconds", type=float, default=None, help="Maximum seconds of audio per inference chunk (default 6.0)")
    parser.add_argument("--disable-internal-vad", action="store_true", help="Force-disable faster-whisper internal VAD (sets vad_filter=False)")
    parser.add_argument("--debug-audio-levels", action="store_true", help="Log per-chunk RMS/peak and voice decision")
    parser.add_argument("--no-output-warning-seconds", type=float, default=30.0, help="Warn if no segments emitted after this many seconds (default 30s; set 0 to disable)")

    args = parser.parse_args()

    # If a specific PulseAudio source is requested, export it before any PyAudio stream opens
    if args.pulse_source:
        os.environ["PULSE_SOURCE"] = args.pulse_source
        print(f"🔊 Using PulseAudio source: {args.pulse_source}")

    # Early exit: list devices
    if args.list_audio_devices:
        try:
            import pyaudio  # type: ignore
        except Exception:
            print("PyAudio not installed; cannot list devices")
            return 1
        pa = pyaudio.PyAudio()
        print("Available audio devices (index | in_ch | out_ch | default_sample_rate | name):")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(f"{i:3d} | {info.get('maxInputChannels',0):3d} | {info.get('maxOutputChannels',0):3d} | {int(info.get('defaultSampleRate',0)):5d} | {info.get('name','Unknown')}")
        pa.terminate()
        return 0

    print("🚀 Whimper - GPU-Accelerated Live Audio Transcription")
    print("=" * 60)
    print(f"🤖 Model: {args.model if not args.model_path else args.model_path}")
    print(f"🌍 Language: {args.language}")
    print(f"⚙️  Device: {args.device}")
    print(f"🔧 Compute: {args.compute_type}")
    if args.device == "cuda" and args.device_index is not None:
        print(f"🧭 GPU Index: {args.device_index}")
    selected_device_index = args.audio_device
    selected_device_name: Optional[str] = None
    if args.audio_device_name is not None and args.audio_device is not None:
        print("⚠️ Both --audio-device and --audio-device-name provided; using explicit index")
    elif args.audio_device_name is not None:
        try:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            name_lower = args.audio_device_name.lower()
            best_idx = None
            best_inputs = -1
            all_names: list[tuple[int,str,int,int]] = []  # (index, name, in_ch, out_ch)
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                nm = str(info.get('name',''))
                all_names.append((i, nm, info.get('maxInputChannels',0), info.get('maxOutputChannels',0)))
                if name_lower in nm.lower() and info.get('maxInputChannels',0) > 0:
                    # prefer highest input channel count
                    if info.get('maxInputChannels',0) > best_inputs:
                        best_idx = i
                        best_inputs = info.get('maxInputChannels',0)
                        selected_device_name = nm
            pa.terminate()
            if best_idx is None:
                if args.suggest_on_fail:
                    # Provide fuzzy suggestions using simple ratio
                    import difflib
                    candidates = [n for _,n,inch,outch in all_names if inch>0]
                    close = difflib.get_close_matches(args.audio_device_name, candidates, n=5, cutoff=0.3)
                    if close:
                        print(f"❌ No exact/substring match for '{args.audio_device_name}'. Did you mean:")
                        for c in close:
                            print(f"   - {c}")
                    else:
                        print(f"❌ No input device matched substring '{args.audio_device_name}' and no close suggestions.")
                else:
                    print(f"❌ No input device matched substring: {args.audio_device_name} (try --suggest-on-fail)")
                return 1
            selected_device_index = best_idx
        except Exception as exc:
            print(f"❌ Failed to resolve audio device name: {exc}")
            return 1

    # Auto-fallback: if no device specified and pulse virtual input exists, use it
    if selected_device_index is None and args.audio_device_name is None and args.audio_device is None:
        try:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            pulse_idx = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                nm = str(info.get('name','')).lower()
                if 'pulse' in nm and info.get('maxInputChannels',0) > 0:
                    pulse_idx = i
                    selected_device_name = info.get('name','pulse')
                    break
            pa.terminate()
            if pulse_idx is not None:
                selected_device_index = pulse_idx
                print(f"🎤 Auto-selected PulseAudio virtual input: {selected_device_index} ({selected_device_name})")
        except Exception:
            pass

    if selected_device_index is not None:
        if selected_device_name:
            print(f"🎤 Audio Device: {selected_device_index} ({selected_device_name})")
        else:
            print(f"🎤 Audio Device: {selected_device_index}")
    else:
        print("🎤 Audio Device: Default")
    print(f"🔈 VAD: {'disabled' if args.no_vad else 'enabled'}")
    if args.min_chunk_seconds is not None:
        print(f"⏱️  Min chunk: {args.min_chunk_seconds:.2f}s")
    if args.max_chunk_seconds is not None:
        print(f"⏱️  Max chunk: {args.max_chunk_seconds:.2f}s")
    if args.disable_internal_vad:
        print("🧪 Internal VAD: disabled (forcing vad_filter=False)")
    if args.debug_audio_levels:
        print("🔬 Debug audio levels: ON")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Collect final segments if export flags are set
        collected: List[TranscriptionResult | str] = []

        def collecting_callback(result: TranscriptionResult | str) -> None:
            # Forward to default printer
            gpu_transcription_callback(result)
            # Collect only final segments (or strings from legacy callback)
            if isinstance(result, TranscriptionResult):
                if result.is_final:
                    collected.append(result)
            else:
                collected.append(result)

        callback_fn: Callable[[TranscriptionResult | str], None]
        if args.out_json or args.out_srt:
            callback_fn = collecting_callback
        else:
            callback_fn = gpu_transcription_callback

        with GPULiveTranscriber(
            model_size=args.model,
            model_path=args.model_path,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            callback=callback_fn,
            use_vad=not args.no_vad,
            device_index=args.device_index,
            min_chunk_seconds=args.min_chunk_seconds,
            max_chunk_seconds=args.max_chunk_seconds,
            disable_internal_vad=args.disable_internal_vad,
            debug_audio_levels=args.debug_audio_levels,
        ) as transcriber:
            transcriber.start_recording(device_index=selected_device_index)
            print("🎧 Listening... speak into your microphone!")
            try:
                last_output_time = time.time()
                warned = False
                while True:
                    time.sleep(0.1)
                    # Warning if no output for too long
                    if args.no_output_warning_seconds and args.no_output_warning_seconds > 0:
                        # Use transcriber's stats: if transcription_count increases, treat as output activity
                        if transcriber.transcription_count > 0:
                            last_output_time = time.time()
                        else:
                            # no outputs yet; keep original start time
                            pass
                        if not warned and (time.time() - last_output_time) > args.no_output_warning_seconds:
                            warned = True
                            logger.warning("⚠️ %ds elapsed with no emitted segments. If you're speaking, try: remove --no-vad, enable --debug-audio-levels, or increase --max-chunk-seconds.", int(args.no_output_warning_seconds))
            except KeyboardInterrupt:
                print("\n🛑 Stopping transcription...\n")
        # After clean exit, write outputs if requested
        if collected:
            if args.out_json:
                _write_segments_json(collected, args.out_json, pretty=args.pretty_json)
                print(f"📝 Wrote JSON segments to {args.out_json}")
            if args.out_srt:
                _write_segments_srt(collected, args.out_srt)
                print(f"📝 Wrote SRT to {args.out_srt}")
    except Exception as exc:
        logger.error("❌ Application error: %s", exc)
        return 1

    return 0


__all__ = [
    "AudioConfig",
    "TranscriptionResult",
    "SimpleVAD",
    "StreamingTranscriptionSession",
    "GPULiveTranscriber",
    "gpu_transcription_callback",
    "main",
]
