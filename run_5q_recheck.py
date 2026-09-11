"""
run_5q_recheck.py — Re-run 5 questions avec max_tokens augmentés + juge sans troncature [:4000].

Objectif : mesurer l'impact des deux corrections :
  1. Synthesizer max_tokens 2500→6000 (DR) / 1000→4000 (VK) → réponses complètes
  2. Juge reçoit la réponse entière (plus d'answer[:4000])

Compare avec les résultats stockés dans le backup principal.
"""
import json, re, sys, time, requests
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import importlib
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

BACKUP   = "comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET_backup_20260717_070832.json"
BASE_URL = "http://localhost:8000/api/query"
HEADERS  = {"Content-Type": "application/json"}
OUT_JSON = f"comparaisons_rag/recheck_5q_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ── 5 questions cibles (excel_row, delta absolu dans les résultats originaux)
TARGET_ROWS = {35, 40, 80, 25, 16}
CONFIGS     = ["v_vanilla_k10", "v_decomp_raptor"]


def judge_full(question, answer, sources, section, subsection):
    """Juge V4.3 SANS troncature [:4000] — c'est la correction clé."""
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : reponse_substantielle_attendue\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"   # ← PAS de [:4000]
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        return result
    except Exception as e:
        return {"judge_error": str(e), "score_global": None}


def call_rag(question, rag_version, k):
    resp = requests.post(
        BASE_URL,
        json={"question": question, "rag_version": rag_version, "k": k},
        headers=HEADERS, timeout=300
    )
    resp.raise_for_status()
    return resp.json()


# ── Charger le backup (résultats originaux)
with open(BACKUP, encoding="utf-8") as f:
    backup = json.load(f)

orig_by_cfg_row = {}
for cfg in CONFIGS:
    orig_by_cfg_row[cfg] = {e["excel_row"]: e for e in backup.get(cfg, [])}

# Identifier les questions cibles
target_questions = []
for e in backup.get("v_vanilla_k10", []):
    if e["excel_row"] in TARGET_ROWS:
        target_questions.append({
            "excel_row": e["excel_row"],
            "question":  e["question"],
            "section":   e["section"],
            "subsection": e["subsection"],
        })
target_questions.sort(key=lambda x: x["excel_row"])

print(f"\n{'='*70}")
print(f"RECHECK — {len(target_questions)} questions × {len(CONFIGS)} configs")
print(f"Correctifs : max_tokens augmentés + juge sans [:4000]")
print(f"{'='*70}")

results = {}  # cfg -> [entries]
for cfg in CONFIGS:
    k = 10 if cfg == "v_vanilla_k10" else 5
    results[cfg] = []
    print(f"\n── Config : {cfg} (k={k})")

    for q in target_questions:
        row = q["excel_row"]
        question = q["question"]
        print(f"  Q{row:3d}  {question[:60]}...")

        # 1. Appel RAG (nouvelles réponses complètes)
        t0 = time.time()
        try:
            data = call_rag(question, cfg, k)
            rag_elapsed = round(time.time() - t0, 1)
            answer = data.get("answer", "")
            raw_sources = data.get("sources", [])
            sources_for_judge = [
                {"content":   s.get("content") or s.get("extrait") or "",
                 "metadata":  s.get("metadata", {}),
                 "source_type": s.get("source_type", ""),
                 "label":     s.get("label", "")}
                for s in raw_sources
            ]
            print(f"        RAG {rag_elapsed}s  →  {len(answer)} chars  ({len(raw_sources)} src)")
        except Exception as e:
            print(f"        RAG ERREUR: {e}")
            results[cfg].append({"excel_row": row, "question": question, "error": str(e)})
            continue

        # 2. Juge V4.3 — réponse complète, sans troncature
        time.sleep(1.0)
        t0 = time.time()
        scores = judge_full(question, answer, sources_for_judge,
                            q["section"], q["subsection"])
        judge_elapsed = round(time.time() - t0, 1)

        sg = scores.get("score_global")
        orig = orig_by_cfg_row[cfg].get(row, {})
        orig_sg = orig.get("score_global")
        delta = (float(sg) - float(orig_sg)) if (sg is not None and orig_sg is not None) else None

        print(f"        Juge {judge_elapsed}s  →  nouveau={sg}  ancien={orig_sg}"
              f"  Δ={delta:+.2f}" if delta is not None else f"  Δ=?")

        entry = {
            "excel_row":    row,
            "question":     question,
            "section":      q["section"],
            "subsection":   q["subsection"],
            "answer_new":   answer,
            "answer_len_new": len(answer),
            "answer_len_old": len(orig.get("answer", "")),
            "rag_elapsed_s": rag_elapsed,
            "judge_elapsed_s": judge_elapsed,
            "sources": sources_for_judge,
            # Nouveaux scores
            **{k2: v for k2, v in scores.items()},
            # Anciens scores (pour comparaison)
            "orig_score_global": orig_sg,
            "orig_pertinence": orig.get("pertinence"),
            "orig_fondement_factuel": orig.get("fondement_factuel"),
            "orig_nuance_incertitude": orig.get("nuance_incertitude"),
            "orig_coherence_qualiquanti": orig.get("coherence_qualiquanti"),
            "orig_answer_len": len(orig.get("answer", "")),
        }
        results[cfg].append(entry)

# ── Sauvegarde JSON
Path("comparaisons_rag").mkdir(exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[SAUVEGARDE] {OUT_JSON}")

# ── Résumé console
print(f"\n{'─'*70}")
print(f"RÉSUMÉ COMPARATIF (ancien → nouveau, juge complet)")
print(f"{'─'*70}")
DIM_KEYS = ["pertinence", "fondement_factuel", "nuance_incertitude",
            "coherence_qualiquanti", "score_global"]

for cfg in CONFIGS:
    entries = [e for e in results[cfg] if "error" not in e]
    if not entries: continue
    print(f"\n  {cfg}")
    for e in entries:
        q_short = e["question"][:55]
        old_sg = e.get("orig_score_global")
        new_sg = e.get("score_global")
        delta = (float(new_sg) - float(old_sg)) if (new_sg and old_sg) else None
        sign = f"{delta:+.2f}" if delta is not None else "  ?  "
        old_len = e.get("answer_len_old", 0)
        new_len = e.get("answer_len_new", 0)
        print(f"    Q{e['excel_row']:3d} | ancien={old_sg} → nouveau={new_sg} [{sign}]"
              f" | len {old_len}→{new_len}ch")

print(f"\nJSON → {OUT_JSON}")
print("Terminé.\n")
