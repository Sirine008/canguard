"""canguard API: detection, diagnosis, metrics and the live stream.

Endpoints
---------
* ``GET  /health``   — service + model + Ollama status
* ``GET  /metrics``  — evaluation results (``reports/metrics.json`` + RAG)
* ``POST /detect``   — score one window of raw CAN messages
* ``POST /diagnose`` — DTC question -> grounded answer + sources
* ``WS   /stream``   — live replay: frames, detections, bus stats, heatmap
* ``/``              — serves the dashboard (``frontend/``)

Run::

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, REPORTS_DIR, WINDOW_SIZE
from src.api.models_runtime import MODEL_VERSIONS, get_runtime
from src.data.features_live import (
    MessageFeaturizer,
    sequence_tensor,
    window_feature_vector,
)

app = FastAPI(title="canguard", version="1.0")

STREAM_TICK_S = 0.1


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RawMessage(BaseModel):
    timestamp: float
    can_id: int | str = Field(description="int, or hex string like '0316'")
    dlc: int = Field(ge=0, le=8)
    data: list[int] = Field(default_factory=list, max_length=8)


class DetectRequest(BaseModel):
    messages: list[RawMessage]
    model: str = "lstm_autoencoder"


class DiagnoseRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    from src.rag.assistant import ollama_available
    models_ready = (PROJECT_ROOT / "models" / "lstm_ae.onnx").exists()
    return {
        "status": "ok",
        "models_ready": models_ready,
        "model_versions": MODEL_VERSIONS,
        "ollama": ollama_available(),
    }


@app.get("/metrics")
def metrics() -> dict:
    out: dict = {}
    detection_path = REPORTS_DIR / "metrics.json"
    rag_path = REPORTS_DIR / "rag_eval.json"
    if detection_path.exists():
        out["detection"] = json.loads(detection_path.read_text(encoding="utf-8"))
    if rag_path.exists():
        out["rag"] = json.loads(rag_path.read_text(encoding="utf-8"))
    if not out:
        raise HTTPException(404, "No evaluation results yet: run "
                                 "src.models.evaluate and src.rag.evaluate_rag")
    return out


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
    if len(request.messages) != WINDOW_SIZE:
        raise HTTPException(
            422, f"expected exactly {WINDOW_SIZE} messages, "
                 f"got {len(request.messages)}")
    featurizer = MessageFeaturizer()
    live = []
    for message in request.messages:
        can_id = (int(message.can_id, 16) if isinstance(message.can_id, str)
                  else int(message.can_id))
        payload = list(message.data) + [0] * (8 - len(message.data))
        live.append(featurizer.feed(
            message.timestamp, can_id, message.dlc, payload))
    features = window_feature_vector(live)
    sequence = sequence_tensor(live)
    try:
        runtime = get_runtime()
    except FileNotFoundError as error:
        raise HTTPException(503, f"models not trained yet: {error}") from error
    try:
        return runtime.score_window(features, sequence, request.model)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/diagnose")
def diagnose(request: DiagnoseRequest) -> dict:
    from src.rag.assistant import answer
    try:
        response = answer(request.question)
    except Exception as error:  # ChromaDB missing / not built yet
        raise HTTPException(
            503, f"knowledge base unavailable ({error}); "
                 "run: python -m src.rag.build_kb") from error
    return {
        "answer": response.answer,
        "sources": response.sources,
        "grounded": response.grounded,
        "llm_used": response.llm_used,
    }


# ---------------------------------------------------------------------------
# WebSocket live stream
# ---------------------------------------------------------------------------

@app.websocket("/stream")
async def stream(socket: WebSocket) -> None:
    from src.simulator.replay import ReplayEngine

    await socket.accept()
    engine = ReplayEngine()
    await socket.send_json({"type": "reset",
                            "status": engine.advance(0.0)["status"]})

    async def control_loop() -> None:
        nonlocal engine
        while True:
            message = json.loads(await socket.receive_text())
            cmd, value = message.get("cmd"), message.get("value")
            if cmd == "play":
                engine.play()
            elif cmd == "pause":
                engine.pause()
            elif cmd == "seek":
                engine.seek(float(value))
            elif cmd == "speed":
                engine.set_speed(float(value))
            elif cmd in ("source", "model"):
                was_playing = engine.playing
                source = value if cmd == "source" else engine.source
                model = value if cmd == "model" else engine.model
                engine = ReplayEngine(source=source, model=model)
                if was_playing:
                    engine.play()
                await socket.send_json(
                    {"type": "reset", "status": engine.advance(0.0)["status"]})

    control = asyncio.create_task(control_loop())
    try:
        while True:
            await asyncio.sleep(STREAM_TICK_S)
            payload = await asyncio.to_thread(engine.advance, STREAM_TICK_S)
            payload["type"] = "tick"
            await socket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        control.cancel()


# Static dashboard - mounted last so API routes win
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"),
                           html=True), name="frontend")
