"""
Telephony audio adapters: decode mu-law/a-law or PCM payloads and resample to 16kHz float32.

Designed for feeding Whimper's StreamingTranscriptionSession via session.add_audio().
"""

from __future__ import annotations

import audioop
from typing import Literal

import numpy as np
import torch
import torchaudio


Encoding = Literal["mulaw", "alaw", "pcm8", "pcm16"]


class TelephonyAdapter:
    """Converts telephony audio payloads to 16kHz mono float32 in [-1, 1].

    - Accepts 8 kHz input for typical PSTN/SIP gateways (G.711 mu-law/a-law or PCM).
    - Uses Python's audioop for G.711 decoding (ulaw2lin/alaw2lin).
    - Uses torchaudio Resample for high-quality resampling to 16 kHz.
    """

    def __init__(self, input_sr: int = 8000, output_sr: int = 16000) -> None:
        self.input_sr = int(input_sr)
        self.output_sr = int(output_sr)
        self._resampler = torchaudio.transforms.Resample(self.input_sr, self.output_sr)

    def decode_and_resample(self, payload: bytes, encoding: Encoding) -> np.ndarray:
        """Decode telephony payload to float32 16k mono suitable for Whisper.

        Args:
            payload: Raw RTP audio bytes
            encoding: 'mulaw' | 'alaw' | 'pcm8' | 'pcm16'
        Returns:
            np.ndarray float32 mono at output_sr in [-1, 1]
        """
        if not payload:
            return np.zeros(0, dtype=np.float32)

        if encoding == "mulaw":
            # Convert mu-law (8-bit) to 16-bit linear PCM
            pcm16 = audioop.ulaw2lin(payload, 2)  # width=2 bytes => int16
            wav = np.frombuffer(pcm16, dtype=np.int16)
        elif encoding == "alaw":
            pcm16 = audioop.alaw2lin(payload, 2)
            wav = np.frombuffer(pcm16, dtype=np.int16)
        elif encoding == "pcm8":
            # 8-bit unsigned PCM (0..255) -> int16
            pcm8 = np.frombuffer(payload, dtype=np.uint8).astype(np.int16)
            wav = ((pcm8 - 128) << 8).astype(np.int16)
        elif encoding == "pcm16":
            wav = np.frombuffer(payload, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported telephony encoding: {encoding}")

        # Normalize to float32 [-1, 1]
        wav_f32 = torch.from_numpy(wav).to(torch.float32) / 32768.0
        wav_f32 = wav_f32.unsqueeze(0)  # [1, T]

        # Resample 8k -> 16k (or input_sr -> output_sr)
        if self.input_sr != self.output_sr:
            wav_f32 = self._resampler(wav_f32)

        wav_f32 = wav_f32.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        return wav_f32
