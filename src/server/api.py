from __future__ import annotations

import base64
import json
from typing import Optional, Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..adapters.telephony_adapter import TelephonyAdapter
from ..infra.session_manager import SessionManager


app = FastAPI(title="Whimper IVR API", version="0.1.0")


class CreateSessionRequest(BaseModel):
    session_id: str


class AddAudioRequest(BaseModel):
    session_id: str
    # Encoded telephony payload (binary) as base64 string for REST; WS will send bytes directly
    payload_b64: str
    encoding: str  # 'mulaw' | 'alaw' | 'pcm8' | 'pcm16'


session_manager = SessionManager(
    gpu_indices=None,  # auto-detect GPUs, fallback to CPU
    model_size="large-v3",
    compute_type="float16",
    max_sessions_per_gpu=5,
    language="en",
    use_vad=True,
    beam_size=1,
)

telephony = TelephonyAdapter(input_sr=8000, output_sr=16000)


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # Results may be list[TranscriptionResult], map to dict
    text = getattr(obj, "text", None)
    start = getattr(obj, "start", None)
    end = getattr(obj, "end", None)
    is_final = getattr(obj, "is_final", None)
    confidence = getattr(obj, "confidence", None)
    if text is not None and start is not None and end is not None:
        return {
            "text": str(text),
            "start": float(start),
            "end": float(end),
            "is_final": bool(is_final) if is_final is not None else True,
            "confidence": float(confidence) if confidence is not None else 1.0,
        }
    return str(obj)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/session")
def create_session(req: CreateSessionRequest):
    try:
        device, gpu = session_manager.create_session(req.session_id)
        return {"ok": True, "device": device, "gpu_index": gpu}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"ok": True}


@app.post("/transcribe")
def add_audio(req: AddAudioRequest):
    try:
        raw = base64.b64decode(req.payload_b64)
        audio = telephony.decode_and_resample(raw, req.encoding)
        session_manager.add_audio(req.session_id, audio)
        out = session_manager.process_next(req.session_id)
        return {"ok": True, "result": _to_jsonable(out)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/{session_id}")
async def ws_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        # Expect a text json control message first: {"encoding": "mulaw|alaw|pcm8|pcm16"}
        init = await websocket.receive_text()
        init_cfg = json.loads(init)
        encoding = init_cfg.get("encoding", "mulaw")
        # Ensure session exists
        try:
            session_manager.create_session(session_id)
        except Exception:
            pass

        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                audio = telephony.decode_and_resample(msg["bytes"], encoding)
                session_manager.add_audio(session_id, audio)
                out = session_manager.process_next(session_id)
                await websocket.send_text(json.dumps({"result": _to_jsonable(out)}))
            elif "text" in msg and msg["text"] == "close":
                break
    except WebSocketDisconnect:
        pass
    finally:
        session_manager.delete_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass
