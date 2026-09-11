"""test_init_only.py — Vérifie que pipeline.init() ne crashe pas."""
import sys, os, pathlib, traceback

# Forcer stdout et stderr sur le même flux pour tout capturer
sys.stderr = sys.stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

os.environ["LLM_BACKEND"]       = "api"
os.environ["PHASE1_STRUCTURED"] = "1"

print("[1] imports de base OK")
sys.stdout.flush()

try:
    import rag_v12_local as v12
    print(f"[2] rag_v12_local importé, _GUARD_AVAILABLE={v12._GUARD_AVAILABLE}")
    sys.stdout.flush()
except Exception:
    print("[2] ERREUR import rag_v12_local:")
    traceback.print_exc()
    sys.exit(1)

try:
    from rag_v12_local import RaptorSubQuestionPipelineV12
    pipeline = RaptorSubQuestionPipelineV12(n_evidence_chunks=5)
    print("[3] Objet créé")
    sys.stdout.flush()
except Exception:
    print("[3] ERREUR création pipeline:")
    traceback.print_exc()
    sys.exit(1)

try:
    print("[4] Appel pipeline.init() — chargement bge-m3 + ChromaDB...")
    sys.stdout.flush()
    pipeline.init()
    print("[5] init() OK — pipeline prêt")
    sys.stdout.flush()
except Exception:
    print("[4/5] ERREUR init():")
    traceback.print_exc()
    sys.exit(1)
