"""
test_local_4cases.py — 4 cas de test local vs API (vis-à-vis).

Exécuter depuis c:\\These\\Données2\\fichiers_pour_rag avec LLM_BACKEND=local.
Nécessite llama-server actif sur localhost:8080 avec :
  --jinja --reasoning-budget 0 --parallel 1

Usage :
  LLM_BACKEND=local PHASE1_STRUCTURED=1 python local/test_local_4cases.py
  python local/test_local_4cases.py  # LLM_BACKEND=local forcé dans le script

Test d'étanchéité réseau :
  Couper le réseau, relancer une question — si ça répond, c'est vraiment local.
"""
import sys, os, io, pathlib, json, time

# UTF-8 output Windows — line_buffering=True pour voir la sortie en temps réel
# même quand stdout est redirigé vers un pipe (Start-Process, harness background).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

os.environ["LLM_BACKEND"]       = "local"
os.environ["PHASE1_STRUCTURED"] = "1"

import rag_v12_local as v12
from rag_v12_local import RaptorSubQuestionPipelineV12

# ── Réponses API de référence ──────────────────────────────────────────────────
API_REF_PATH = ROOT / "comparaisons_rag" / "rapport_guard_sources.json"
try:
    with open(API_REF_PATH, encoding="utf-8") as f:
        _api_ref = json.load(f)
    API_REF = {c["label"]: c for c in _api_ref}
except Exception as e:
    print(f"[WARN] Impossible de charger les réponses API de référence : {e}")
    API_REF = {}

SEP  = "=" * 76
SEP2 = "-" * 76

CASES = [
    {
        "label":    "Aiti — bien-etre",
        "question": "Quel est le niveau de bien-être subjectif à Aïti ?",
        "communes": ["Aiti"],
        "expected": "OppChoVec cité, absence verbatims explicite, aucun énoncé qualitatif propre à Aïti",
    },
    {
        "label":    "Piedicorte — qualite de vie",
        "question": "Comment les habitants de Piedicorte-di-Gaggio perçoivent-ils leur qualité de vie ?",
        "communes": ["Piedicorte-di-Gaggio"],
        "expected": "Scores OppChoVec, absence verbatims explicite",
    },
    {
        "label":    "Ajaccio — bien-etre",
        "question": "Quel est le niveau de bien-être subjectif à Ajaccio ?",
        "communes": ["Ajaccio"],
        "expected": "Réponse riche, silence_map vide, guard PASS",
    },
    {
        "label":    "Aiti — verbatims scope_reel",
        "question": "Quels sont les verbatims citoyens les plus représentatifs à Aïti ?",
        "communes": ["Aiti"],
        "expected": "Label ⚠ DONNÉES CORSE ENTIÈRE, aucun verbatim présenté comme venant d'Aïti",
    },
]

OUT_DIR = ROOT / "comparaisons_rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_step(metrics: dict, step: str) -> str:
    s = metrics.get("steps", {}).get(step, {})
    return (f"{s.get('elapsed_s', 0):.1f}s "
            f"[p={s.get('prompt_tokens',0)} c={s.get('completion_tokens',0)} "
            f"×{s.get('n_calls',1)}]")


def run_case(pipeline, case: dict) -> dict:
    label    = case["label"]
    question = case["question"]

    print(f"\n{SEP}")
    print(f"  CAS : {label}")
    print(f"  Q   : {question}")
    print(SEP)

    t0     = time.time()
    result = pipeline.query(question)
    elapsed = time.time() - t0

    final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics = result

    guard        = metrics.get("guard", {})
    silence_map  = guard.get("silence_map", {})
    silence_list = guard.get("silence_list", [])
    violations   = guard.get("violations", [])

    # ── Métriques par étape ──────────────────────────────────────
    print(f"\n>> Latence totale     : {elapsed:.1f}s")
    print(f"   décomposition     : {fmt_step(metrics, 'decompose')}")
    print(f"   réponses SQ       : {fmt_step(metrics, 'answer')}")
    print(f"   synthèse          : {fmt_step(metrics, 'synthesize')}")

    # ── Guard ────────────────────────────────────────────────────
    print(f"\n>> silence_map        : {silence_map}")
    print(f">> guard.passed       : {guard.get('passed')}")
    print(f">> n_violations       : {guard.get('n_violations', 0)}")
    if violations:
        for v in violations:
            print(f"   [{v.get('rule','?').upper()}] {v.get('detail','')}")
            print(f"    Evidence : {str(v.get('evidence',''))[:120]}")
    else:
        print("   Aucune violation")

    # ── Réponse locale ───────────────────────────────────────────
    print(f"\n{SEP2}")
    print("RÉPONSE LOCALE :")
    print(SEP2)
    print(final_answer)

    # ── Réponse API (référence) ──────────────────────────────────
    api_case = API_REF.get(label)
    if api_case:
        api_answer   = api_case.get("answer", "")
        api_guard    = api_case.get("guard", {})
        api_elapsed  = api_case.get("elapsed", 0)
        print(f"\n{SEP2}")
        print(f"RÉPONSE API ({api_case.get('label','?')}) — {api_elapsed}s :")
        print(SEP2)
        print(api_answer)
        print(f"\n>> guard API : passed={api_guard.get('passed')}  "
              f"n_violations={api_guard.get('n_violations',0)}  "
              f"silence_map={api_guard.get('silence_map',{})}")
    else:
        print(f"\n[INFO] Pas de réponse API de référence pour '{label}'")

    return {
        "label":        label,
        "question":     question,
        "elapsed":      round(elapsed, 1),
        "scope":        scope,
        "question_type": question_type,
        "guard":        guard,
        "answer":       final_answer,
        "metrics_steps": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in metrics.get("steps", {}).items()
        },
        "sub_qa":  [{"q": e["question"], "a": e["answer"]} for e in (sub_qa or [])],
        "api_ref": {
            "answer":   api_case.get("answer", "") if api_case else "",
            "elapsed":  api_case.get("elapsed", 0) if api_case else 0,
            "guard":    api_case.get("guard",  {}) if api_case else {},
        },
    }


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  test_local_4cases.py — LLM_BACKEND=local, PHASE1_STRUCTURED=1")
    print(f"  llama-server : {v12.LLAMA_BASE_URL}")
    print(f"  modèle       : {v12.LLAMA_MODEL}")
    print(SEP)

    print("\nChargement pipeline...")
    t_load = time.time()
    pipeline = RaptorSubQuestionPipelineV12(n_evidence_chunks=5)
    pipeline.init()
    print(f"Pipeline prêt en {(time.time()-t_load):.0f}s\n")

    results = []
    for case in CASES:
        try:
            r = run_case(pipeline, case)
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"\n[ERREUR] {case['label']} : {exc}")
            traceback.print_exc()
            results.append({"label": case["label"], "error": str(exc)})

    # Sauvegarder
    out_path = OUT_DIR / "local_4cases_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n{SEP}")
    print(f"  Résultats sauvegardés : {out_path}")
    print(SEP)

    # Résumé final
    print(f"\n{'Label':<35} {'Local (s)':>10} {'API (s)':>8} {'Guard local':>14} {'Guard API':>12}")
    print("-" * 82)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<33} ERREUR")
            continue
        api_passed = r.get("api_ref", {}).get("guard", {}).get("passed")
        loc_passed = r.get("guard", {}).get("passed")
        loc_viol   = r.get("guard", {}).get("n_violations", "?")
        api_viol   = r.get("api_ref", {}).get("guard", {}).get("n_violations", "?")
        print(f"  {r['label']:<33} {r['elapsed']:>9.1f}s "
              f"{r['api_ref'].get('elapsed',0):>7.1f}s "
              f"  {'PASS' if loc_passed else f'FAIL({loc_viol}viol)':<14}"
              f"  {'PASS' if api_passed else f'FAIL({api_viol}viol)'}")
