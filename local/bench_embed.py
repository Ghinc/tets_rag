"""
bench_embed.py — Mesure latence embedding bge-m3 CPU (une requête).

Usage :
  python local/bench_embed.py

Aucun appel LLM — mesure purement locale.
"""
import os, sys, time
os.environ.setdefault("LLM_BACKEND", "api")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "local"))

from rag_v12_local import RaptorSubQuestionPipelineV12

QUESTION = "Quels sont les problèmes de logement à Ajaccio ?"
N_WARMUP = 1
N_BENCH  = 5

print("Chargement pipeline (bge-m3 → CPU)...")
t_load = time.perf_counter()
p = RaptorSubQuestionPipelineV12()
p.init()
dt_load = (time.perf_counter() - t_load) * 1000
print(f"Chargement : {dt_load:.0f} ms\n")

encode = p.retriever._encode_query

print(f"Warmup ({N_WARMUP} appel)...")
for _ in range(N_WARMUP):
    encode(QUESTION)

print(f"Bench ({N_BENCH} appels)...")
times_ms = []
for i in range(N_BENCH):
    t0 = time.perf_counter()
    encode(QUESTION)
    times_ms.append((time.perf_counter() - t0) * 1000)
    print(f"  [{i+1}] {times_ms[-1]:.0f} ms")

moy = sum(times_ms) / N_BENCH
med = sorted(times_ms)[N_BENCH // 2]
print(f"\nbge-m3 CPU (PyTorch) — latence embedding requête :")
print(f"  Moyenne : {moy:.0f} ms")
print(f"  Médiane : {med:.0f} ms")
print(f"  Min / Max : {min(times_ms):.0f} / {max(times_ms):.0f} ms")
print(f"\n(Cible brief : ~80 ms sur 14 cœurs)")
print(f"(Pour ONNX int8 : pip install optimum[onnxruntime] puis backend='onnx', dtype='int8')")
