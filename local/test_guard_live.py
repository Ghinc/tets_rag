"""
test_guard_live.py — Vérifie en conditions réelles :
  1. Le guard est disponible (_GUARD_AVAILABLE=True)
  2. La capture synth_context n'est pas vide (affiché par [GUARD][synth])
  3. Le guard se déclenche sur Aïti (entité silencée, aucune donnée)

Question test : "Quel est le niveau de bien-être subjectif à Aïti ?"
Mode : PHASE1_STRUCTURED=1 (extraction structurée — déclenche le silence guard)
"""
import sys, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

os.environ["LLM_BACKEND"]       = "api"
os.environ["PHASE1_STRUCTURED"] = "1"

import rag_v12_local as v12
print(f"[TEST] _GUARD_AVAILABLE = {v12._GUARD_AVAILABLE}")

from rag_v12_local import RaptorSubQuestionPipelineV12

print("[TEST] Chargement pipeline...")
pipeline = RaptorSubQuestionPipelineV12(n_evidence_chunks=5)
pipeline.init()
print("[TEST] Pipeline prêt\n")

question = "Quel est le niveau de bien-être subjectif à Aïti ?"
print(f"[TEST] Question : {question}\n")

result = pipeline.query(question)
final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics = result

print(f"\n[TEST] Réponse ({len(final_answer)} chars)")
print(f"[TEST] sources récupérées : {len(sources or [])}")
guard = metrics.get("guard", {})
print(f"[TEST] guard.available  = {guard.get('available')}")
print(f"[TEST] guard.passed     = {guard.get('passed')}")
print(f"[TEST] guard.n_violations = {guard.get('n_violations')}")
print(f"[TEST] silence_list     = {guard.get('silence_list')}")
for v in guard.get("violations", []):
    print(f"  [{v.get('rule','?').upper()}] {v.get('detail','')}")
    print(f"    evidence: {str(v.get('evidence',''))[:100]}")
