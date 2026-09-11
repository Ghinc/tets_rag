"""
embedder_service.py — Service HTTP persistant pour bge-m3 (et bge-reranker optionnel).

Chargement unique au démarrage. Les pipelines se connectent via HTTP au lieu de
recharger les modèles à chaque process (évite 30s de cold-start + conflits DLL).

Usage (lancer depuis le répertoire du projet, terminal visible) :
    python local/embedder_service.py [--port 8765] [--no-reranker]

Endpoints :
    GET  /health    → {"status": "ok", "models": ["bge-m3", ...]}
    POST /encode    {"texts": [...], "batch_size": 32, "normalize": true}
                    → {"embeddings": [[...], ...], "n": N, "dim": D}
"""
import argparse
import sys
import os
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Forcer CPU avant tout import sentence_transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


app = FastAPI(title="EmbedderService", version="1.0.0")

_MODELS: dict = {}   # "bge-m3" → SentenceTransformer instance


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _load_models():
    global _MODELS
    from sentence_transformers import SentenceTransformer

    t = time.time()
    print("[service] Chargement bge-m3 (device=cpu)…", flush=True)
    _MODELS["bge-m3"] = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print(f"[service] bge-m3 prêt en {time.time()-t:.1f}s", flush=True)


# ── Schémas ───────────────────────────────────────────────────────────────────

class EncodeRequest(BaseModel):
    texts: List[str]
    batch_size: int = 32
    normalize: bool = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": list(_MODELS.keys())}


@app.post("/encode")
def encode(req: EncodeRequest):
    if "bge-m3" not in _MODELS:
        raise HTTPException(503, "bge-m3 non disponible")
    if not req.texts:
        raise HTTPException(400, "texts vide")

    embs = _MODELS["bge-m3"].encode(
        req.texts,
        batch_size=req.batch_size,
        normalize_embeddings=req.normalize,
        show_progress_bar=False,
    )
    arr = np.array(embs)
    return {
        "embeddings": arr.tolist(),
        "n": arr.shape[0],
        "dim": arr.shape[1],
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Service d'embedding bge-m3 persistant")
    parser.add_argument("--port", type=int, default=8765, help="Port HTTP (défaut : 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (défaut : 127.0.0.1)")
    args = parser.parse_args()

    print(f"[service] Démarrage sur http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
