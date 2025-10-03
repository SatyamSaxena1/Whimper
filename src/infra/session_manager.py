from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gpu_worker import GPUWorker
import torch


class SessionManager:
    def __init__(
        self,
        gpu_indices: Optional[List[int]] = None,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        max_sessions_per_gpu: int = 5,
        language: str = "en",
        use_vad: bool = True,
        beam_size: int = 1,
    ) -> None:
        self._lock = threading.RLock()
        if gpu_indices is None:
            gpu_indices = list(range(self._detect_gpu_count()))

        self.workers: List[GPUWorker] = []
        device = "cuda" if torch.cuda.is_available() and len(gpu_indices) > 0 else "cpu"
        if device == "cuda":
            for i in gpu_indices:
                self.workers.append(
                    GPUWorker(
                        gpu_index=i,
                        model_size=model_size,
                        compute_type=compute_type,
                        device=device,
                        max_sessions=max_sessions_per_gpu,
                        language=language,
                        use_vad=use_vad,
                        beam_size=beam_size,
                    )
                )
        else:
            # Single CPU worker
            self.workers.append(
                GPUWorker(
                    gpu_index=0,
                    model_size=model_size,
                    compute_type="int8" if compute_type == "float16" else compute_type,
                    device="cpu",
                    max_sessions=max_sessions_per_gpu,
                    language=language,
                    use_vad=use_vad,
                    beam_size=beam_size,
                )
            )
        self._routing: Dict[str, GPUWorker] = {}

    def _detect_gpu_count(self) -> int:
        try:
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            return 0

    def _least_loaded_worker(self) -> GPUWorker:
        # Prefer workers with capacity
        candidates = [w for w in self.workers if w.has_capacity()]
        if not candidates:
            # fall back to the absolute least loaded to raise capacity error downstream
            return min(self.workers, key=lambda w: w.session_count())
        return min(candidates, key=lambda w: w.session_count())

    def create_session(self, session_id: str) -> Tuple[str, int]:
        with self._lock:
            if session_id in self._routing:
                raise ValueError(f"Session exists: {session_id}")
            worker = self._least_loaded_worker()
            worker.create_session(session_id)
            self._routing[session_id] = worker
            return (worker.device, worker.gpu_index)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            worker = self._routing.pop(session_id, None)
        if worker:
            worker.delete_session(session_id)

    def add_audio(self, session_id: str, audio_f32_mono_16k: np.ndarray) -> None:
        worker = self._routing.get(session_id)
        if not worker:
            raise KeyError(f"Unknown session_id: {session_id}")
        worker.add_audio(session_id, audio_f32_mono_16k)

    def process_next(self, session_id: str):
        worker = self._routing.get(session_id)
        if not worker:
            raise KeyError(f"Unknown session_id: {session_id}")
        return worker.process_next(session_id)
