"""
observatory/server.py
=====================
FastAPI host for *Dumè GPT* — the Corsica well-being observation assistant.

Routes
------
GET  /                     the single-page UI
GET  /api/health           pipeline loading status
GET  /api/examples         starter questions (English)
GET  /api/communes         all 360 Corsican communes
GET  /api/commune?name=&view=  one commune's sidebar profile (view: objective|subjective)
GET  /api/config?code=...  dev-console unlock + choice lists (code-gated)
POST /api/ask              ask a question — Server-Sent Events stream:
                             event: progress  {stage, i?, n?}
                             event: result    {answer, sources, meta, ...}
                             event: error     {message, busy?}

Run:  .venv\\Scripts\\python.exe local\\observatory\\server.py
      (or local\\observatory\\run_observatory.ps1)

Environment
-----------
OBS_PORT   8600 | OBS_HOST 127.0.0.1 | OBS_DEV_CODE "corse2026"
OBS_FORCE_CPU_EMBED "1" | OBS_ENABLE_V11 "1"
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import queue
import sys
import threading
from contextlib import asynccontextmanager

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pipeline_manager as pm  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
import uvicorn  # noqa: E402

DEV_CODE = os.getenv("OBS_DEV_CODE", "corse2026")
UI_FILE = os.path.join(_HERE, "ui", "index.html")
DASHBOARD_DIR = os.path.join(_HERE, "dashboard")

EXAMPLE_QUESTIONS = [
    "How do residents of Ajaccio describe their quality of life?",
    "Which aspects of well-being are rated lowest in Bastia?",
    "Is there a gap between objective indicators and residents' perceptions in Corte?",
    "What concerns come up most often about daily life in Corsica's rural communes?",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pm.start_loading()
    yield


app = FastAPI(title="Dumè GPT", version="1.0.0", lifespan=lifespan)

# OppChoVec dashboard (Leaflet maps, LISA/CAH clustering, correlations,
# parangons...) — a separate static app (its own repo: oppchovec_visu),
# copied wholesale into local/observatory/dashboard/ and served as-is.
# html=True lets "/dashboard/" resolve to its index.html and every relative
# asset request it makes (script.js, data JSON, geojson) resolve correctly —
# see local/observatory/dashboard/.gitignore for what's tracked vs disk-only.
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


# --------------------------------------------------------------------------- #
# Static / meta
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/api/health")
def health():
    return pm.status()


@app.get("/api/examples")
def examples():
    return {"questions": EXAMPLE_QUESTIONS}


@app.get("/api/communes")
def communes():
    return {"communes": pm.commune_names()}


@app.get("/api/commune")
def commune(name: str = "", view: str = "objective"):
    return pm.commune_profile(name, "subjective" if view == "subjective" else "objective") or {}


@app.get("/api/config")
def config(code: str = ""):
    if code != DEV_CODE:
        return JSONResponse({"ok": False})
    return {
        "ok": True,
        "roles": pm.ROLES,
        "defaults": pm.DEFAULTS,
        "reference": pm.REFERENCE_PIPELINE,
        "provider_models": pm.PROVIDER_MODELS,
        "output_languages": pm.OUTPUT_LANGUAGES,
        "versions": pm.VERSION_CHOICES,
    }


# --------------------------------------------------------------------------- #
# Config resolution — dev overrides honoured only with the correct code.
# `commune` is a normal user feature and is always honoured.
# --------------------------------------------------------------------------- #
def _valid_role(r) -> bool:
    return (isinstance(r, dict)
            and r.get("provider") in pm.PROVIDER_MODELS
            and r.get("model") in pm.PROVIDER_MODELS[r["provider"]])


def _resolve_config(raw: dict) -> dict:
    cfg = copy.deepcopy(pm.DEFAULTS)
    raw = raw if isinstance(raw, dict) else {}

    commune = (raw.get("commune") or "").strip()
    if commune and commune.lower() not in ("all", "corsica", "all of corsica"):
        by_lower = {c.lower(): c for c in pm.commune_names()}
        if commune.lower() in by_lower:
            cfg["commune"] = by_lower[commune.lower()]

    if raw.get("output_language") in pm.OUTPUT_LANGUAGES:
        cfg["output_language"] = raw["output_language"]

    if raw.get("dev_code") != DEV_CODE:
        return cfg

    if raw.get("version") in pm.VERSION_CHOICES:
        cfg["version"] = raw["version"]

    for role in pm.ROLES:
        if _valid_role(raw.get(role)):
            cfg[role] = {"provider": raw[role]["provider"], "model": raw[role]["model"]}

    try:
        if raw.get("k") is not None:
            cfg["k"] = max(1, min(20, int(raw["k"])))
    except (TypeError, ValueError):
        pass
    try:
        if raw.get("n_subquestions") is not None:
            cfg["n_subquestions"] = max(1, min(8, int(raw["n_subquestions"])))
    except (TypeError, ValueError):
        pass
    temp = raw.get("temperature")
    if temp not in (None, ""):
        try:
            cfg["temperature"] = max(0.0, min(2.0, float(temp)))
        except (TypeError, ValueError):
            pass
    return cfg


# --------------------------------------------------------------------------- #
# Ask — SSE
# --------------------------------------------------------------------------- #
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask")
async def ask(req: Request):
    body = await req.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    if len(question) > 2000:
        question = question[:2000]

    cfg = _resolve_config(body.get("config") or {})
    q: "queue.Queue" = queue.Queue()

    def emit(ev: dict) -> None:
        q.put(("progress", ev))

    def worker() -> None:
        try:
            result = pm.run(question, cfg, emit)
            q.put(("result", result))
        except pm.Busy as exc:
            q.put(("error", {"message": str(exc), "busy": True}))
        except pm.NotReady as exc:
            q.put(("error", {"message": str(exc), "loading": True}))
        except Exception as exc:                                   # noqa: BLE001
            q.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            q.put(("__done__", None))

    threading.Thread(target=worker, name="obs-query", daemon=True).start()

    async def stream():
        loop = asyncio.get_event_loop()
        yield _sse("progress", {"stage": "received"})
        while True:
            kind, data = await loop.run_in_executor(None, q.get)
            if kind == "__done__":
                break
            yield _sse(kind, data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    host = os.getenv("OBS_HOST", "127.0.0.1")
    port = int(os.getenv("OBS_PORT", "8600"))
    print("=" * 64)
    print("  Dumè GPT — Corsica well-being observation assistant")
    print(f"  UI         : http://{host}:{port}/")
    print(f"  Dev switch : http://{host}:{port}/?dev={DEV_CODE}")
    print("  (the RAPTOR pipeline loads in the background — the first")
    print("   question waits until /api/health reports ready)")
    print("=" * 64)
    uvicorn.run(app, host=host, port=port, log_level="info")
