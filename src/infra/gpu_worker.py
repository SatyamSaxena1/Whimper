"""
GPU worker hosting a single Whisper model instance to serve multiple concurrent
StreamingTranscriptionSession objects (one per IVR call).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
from faster_whisper import WhisperModel

from ..whimper import StreamingTranscriptionSession


class GPUWorker:
    def __init__(
        self,
        gpu_index: int,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        device: str = "cuda",
        max_sessions: int = 5,
        language: str = "en",
        use_vad: bool = True,
        beam_size: int = 1,
    ) -> None:
        self.gpu_index = int(gpu_index)
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.max_sessions = int(max_sessions)
        self.language = language
        self.use_vad = use_vad
        self.beam_size = int(beam_size)

        self._sessions: Dict[str, StreamingTranscriptionSession] = {}
        self._lock = threading.RLock()

        # Load model once per worker (GPU or CPU)
        kwargs = dict(
            device=self.device,
            compute_type=self.compute_type,
        )
        if self.device == "cuda":
            kwargs["device_index"] = self.gpu_index
        self.model = WhisperModel(self.model_size, **kwargs)
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros(16000, dtype=np.float32)
        # Single quick pass to allocate kernels/caches
        self.model.transcribe(dummy, beam_size=1)

    def create_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session exists: {session_id}")
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("GPU at capacity")
            sess = StreamingTranscriptionSession(
                model=self.model,
                language=self.language,
                use_vad=self.use_vad,
                beam_size=self.beam_size,
            )
            # IVR-optimized latency knobs
            sess.MIN_CHUNK_SECONDS = 0.30
            sess.MAX_CHUNK_SECONDS = 2.5
            sess.SILENCE_STEP_SECONDS = 0.25
            sess.SAME_OUTPUT_THRESHOLD = 2
            self._sessions[session_id] = sess

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def add_audio(self, session_id: str, audio_f32_mono_16k: np.ndarray) -> None:
        sess = self._sessions.get(session_id)
        if not sess:
            raise KeyError(f"Unknown session_id: {session_id}")
        sess.add_audio(audio_f32_mono_16k)

    def process_next(self, session_id: str):
        sess = self._sessions.get(session_id)
        if not sess:
            raise KeyError(f"Unknown session_id: {session_id}")
        return sess.process_next()

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def has_capacity(self) -> bool:
        with self._lock:
            return len(self._sessions) < self.max_sessions

    def at_capacity(self) -> bool:
        return not self.has_capacity()
