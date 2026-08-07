"""
run_no_typing_q104_109.py — Complète v_decomp_no_typing avec Q104–Q109 (Limites architecturales).

Ces 6 questions n'existent pas dans l'Excel — elles sont hardcodées ici (cf. add_limites_arch_questions.py).
Juge V4.3 pour rester cohérent avec les 103 autres questions du run no_typing.

Usage :
    python run_no_typing_q104_109.py
"""

import json, re, sys, time, requests, shutil
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

# ── Constantes ──────────────────────────────────────────────────────────────
API_BASE     = "http://localhost:8000/api/query"
RAG_VERSION  = "v_decomp_no_typing"
K            = 5
OUT_DIR      = Path("comparaisons_rag")
CKPT_JSONL   = OUT_DIR / "no_typing_checkpoint.jsonl"
FINAL_JSON   = OUT_DIR / "no_typing_109q_FINAL.json"
COMPLET_JSON = OUT_DIR / "ablations_103q_v43_gpt4o_COMPLET.json"

NEW_QUESTIONS = [
    {
        "excel_row": 104,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Pour quelles communes corses peut-on croiser à la fois des données d'enquête citoyenne, des entretiens semi-directifs, et un score OppChoVec complet ?",
    },
    {
        "excel_row": 105,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Classer les 10 communes corses ayant les écarts les plus marqués entre indicateurs objectifs et perceptions subjectives.",
    },
    {
        "excel_row": 106,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Quelles sont les communes corses qui présentent un score Vec inférieur à 3/10 et où les habitants expriment néanmoins une satisfaction élevée concernant leur cadre de vie ?",
    },
    {
        "excel_row": 107,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Existe-t-il une corrélation entre la qualité du logement et la satisfaction des habitants à l'échelle des communes corses ?",
    },
    {
        "excel_row": 108,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Sur l'ensemble du corpus, comment l'âge des répondants influence-t-il leur perception du bien-être territorial ?",
    },
    {
        "excel_row": 109,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Dans quelle commune les 18-25 ans se sentent-ils le mieux ?",
    },
]


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_done_rows() -> set:
    done = set()
    if CKPT_JSONL.exists():
        for line in CKPT_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    done.add(e["excel_row"])
                except Exception:
                    pass
    return done


def append_checkpoint(entry: dict):
    with open(CKPT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Appels API ───────────────────────────────────────────────────────────────

def call_rag(question: str) -> dict:
    r = requests.post(API_BASE,
                      json={"question": question, "rag_version": RAG_VERSION, "k": K},
                      timeout=300)
    r.raise_for_status()
    return r.json()


def call_judge(question, answer, sources, section, subsection) -> dict:
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : limite_architecturale\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    t0 = time.time()
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        result["judge_elapsed_s"] = round(time.time() - t0, 1)
        return result
    except Exception as e:
        return {"judge_error": str(e), "score_global": None,
                "judge_elapsed_s": round(time.time() - t0, 1)}


# ── Mise à jour COMPLET.json ──────────────────────────────────────────────────

def merge_into_complet(new_entries: list):
    with open(COMPLET_JSON, encoding="utf-8") as f:
        data = json.load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = COMPLET_JSON.with_name(f"ablations_103q_v43_gpt4o_COMPLET_backup_{ts}.json")
    shutil.copy(COMPLET_JSON, backup)
    print(f"Backup → {backup.name}")

    existing_rows = {e["excel_row"] for e in data.get("v_decomp_no_typing", [])}
    added = 0
    for e in new_entries:
        if e["excel_row"] not in existing_rows:
            data.setdefault("v_decomp_no_typing", []).append(e)
            added += 1

    with open(COMPLET_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"COMPLET.json mis à jour : +{added} entrées")
    for ver in data:
        print(f"  {ver}: {len(data[ver])} entrées")
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

done_rows = load_done_rows()
to_run = [q for q in NEW_QUESTIONS if q["excel_row"] not in done_rows]

print(f"Q104–Q109 pour {RAG_VERSION}")
print(f"  Déjà faites : {sorted(done_rows & {q['excel_row'] for q in NEW_QUESTIONS})}")
print(f"  À traiter   : {[q['excel_row'] for q in to_run]}")

new_entries = []

for q_meta in to_run:
    row = q_meta["excel_row"]
    question = q_meta["question"]
    section = q_meta["section"]
    subsection = q_meta["subsection"]

    print(f"\nQ{row:03d} — {question[:70]}")
    print("  RAG...", end="", flush=True)

    t0 = time.time()
    try:
        rag = call_rag(question)
        elapsed_rag = round(time.time() - t0, 1)
        answer  = rag.get("answer", "")
        sources = rag.get("sources", [])
        n_sources = len(sources)
        n_subq    = len(rag.get("sub_questions", []))
        rag_status = "ok"
        print(f" OK ({elapsed_rag}s, {n_sources} sources)", end="", flush=True)
    except Exception as e:
        elapsed_rag = round(time.time() - t0, 1)
        print(f" ERREUR: {e}")
        entry = {
            **q_meta,
            "rag_status": "error", "rag_error": str(e),
            "answer": "", "sources": [], "n_sources": 0,
            "n_subquestions": 0, "rag_elapsed_s": elapsed_rag,
            "score_global": None,
        }
        append_checkpoint(entry)
        new_entries.append(entry)
        continue

    print(" — Judge...", end="", flush=True)
    time.sleep(1.0)

    j = call_judge(question, answer, sources, section, subsection)

    sg  = j.get("score_global")
    mis = j.get("mislabelling_detecte") or {}
    has_mis = any(
        str(v).lower() not in ("non", "false", "", "null", "none", "0")
        for v in mis.values()
    ) if mis else False

    entry = {
        **q_meta,
        "rag_status": rag_status,
        "answer": answer,
        "n_sources": n_sources,
        "n_subquestions": n_subq,
        "rag_elapsed_s": elapsed_rag,
        "sources": sources,
        "pertinence":           j.get("pertinence"),
        "fondement_factuel":    j.get("fondement_factuel"),
        "nuance_incertitude":   j.get("nuance_incertitude"),
        "coherence_qualiquanti": j.get("coherence_qualiquanti"),
        "score_global":         sg,
        "mislabelling_flag":    has_mis,
        "mislabelling_detecte": mis,
        "raisonnement_v43":     j.get("raisonnement_v43", ""),
        "judge_elapsed_s":      j.get("judge_elapsed_s", 0),
        "judge_error":          j.get("judge_error"),
    }

    append_checkpoint(entry)
    new_entries.append(entry)

    flag_str = "✗MIS" if has_mis else "✓"
    print(f" {sg:.2f} {flag_str}" if sg else f" judge_error={j.get('judge_error','?')}")

# ── Mise à jour FINAL.json ────────────────────────────────────────────────────
print("\n── Mise à jour des fichiers ──")

all_entries = []
if CKPT_JSONL.exists():
    for line in CKPT_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                all_entries.append(json.loads(line))
            except Exception:
                pass

all_entries.sort(key=lambda e: e["excel_row"])
with open(FINAL_JSON, "w", encoding="utf-8") as f:
    json.dump({RAG_VERSION: all_entries}, f, ensure_ascii=False, indent=2)
print(f"FINAL.json → {len(all_entries)} entrées")

# ── Mise à jour COMPLET.json ──────────────────────────────────────────────────
merge_into_complet(new_entries)
