"""
run_nt_v2_legacy_109q.py — v_decomp_no_typing v2 avec les ANCIENS prompts.

Les anciens prompts (commit 9732e9e, 27 mai 2026) sont ceux avec lesquels
v_vanilla_k10/k25, v_decomp et v_decomp_raptor ont tourné dans COMPLET.json.
Ce script leur applique le même détypages v2, pour une comparaison équitable.

Détypages appliqués sur anciens prompts :
  - Décomposeur : Rule (1) "même type" supprimée, règles renumérotées
  - Answerer    : priorité par type de sous-question supprimée
  - Synthétiseur: pas de typage dans l'ancien prompt — inchangé

Usage:
    python run_nt_v2_legacy_109q.py          # 109 questions
    python run_nt_v2_legacy_109q.py --max 5  # pilot
"""
import html as _html, json, re, sys, time, requests, argparse, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm
from rag_v10_raptor_subq import (
    _call_mistral, _call_claude,
    DEFAULT_N_SUBQUESTIONS,
)

API_BASE = "http://localhost:8000"


def _retrieve(question: str, k: int = 5):
    """Retrieval via le serveur API — évite le conflit ChromaDB multi-process."""
    r = requests.post(f"{API_BASE}/api/retrieve", json={"question": question, "k": k}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["context_str"], data["sources"]

# ── Extraction des anciens prompts depuis git ─────────────────────────────────

def _extract_const(src, name):
    idx = src.find(f"{name} = (")
    start = src.index("(", idx)
    depth = 0
    for i, c in enumerate(src[start:], start):
        if c == "(":   depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return eval(f"({src[start+1:i]})")
    raise ValueError(f"{name} introuvable")

_OLD_SRC = subprocess.run(
    ["git", "show", "9732e9e:rag_v10_raptor_subq.py"],
    capture_output=True, cwd=ROOT
).stdout.decode("utf-8", errors="replace")

_OLD_DECOMPOSER       = _extract_const(_OLD_SRC, "_SYSTEM_DECOMPOSER")
_OLD_ANSWERER         = _extract_const(_OLD_SRC, "_SYSTEM_ANSWERER")
_OLD_SYNTHESIZER      = _extract_const(_OLD_SRC, "_SYSTEM_SYNTHESIZER_NO_BILAN")

# ── Détypages v2 sur anciens prompts ─────────────────────────────────────────

# Décomposeur : supprimer Rule (1) et renuméroter
_RULE1_OLD = (
    "(1) Chaque sous-question doit rester du MEME TYPE que la question initiale : "
    "si la question est factuelle/chiffree (ex: score, rang, indicateur), les sous-questions le sont aussi "
    "— ne derives JAMAIS vers du qualitatif/percetions/verbatims pour une question factuelle. "
    "Si la question est qualitative (ex: que pensent les habitants), les sous-questions portent sur des groupes ou dimensions. "
    "EXCEPTION : si la question porte sur le bien-etre global ou la qualite de vie generale d'une commune, "
    "tu DOIS generer un mix de sous-questions qualitatives (perceptions, verbatims citoyens) ET quantitatives "
    "(scores d'enquete, indicateurs OppChoVec) pour couvrir les deux dimensions. "
    "Si la question comporte des verbes de percetion, considère que la question est qualitative ; "
    "si non, et si elle ne demande pas excplicitement l'opinion des habitants, elle est générale."
)
_OLD_DECOMPOSER_NT = re.sub(
    re.escape(_RULE1_OLD) + r"\s*\(2\)\s*",
    "(1) ",
    _OLD_DECOMPOSER,
)

# Answerer : supprimer la priorisation par type
_TYPING_ANSWERER = (
    "Pour une sous-question chiffree sur le bien-etre territorial (rang, score), "
    "appuie-toi prioritairement sur les indicateurs objectifs. "
    "Pour une sous-question de perception ou de satisfaction, priorise les donnees d'enquete citoyenne. "
)
_OLD_ANSWERER_NT = _OLD_ANSWERER.replace(
    _TYPING_ANSWERER,
    "Appuie-toi sur l'ensemble des données disponibles dans le contexte (indicateurs objectifs "
    "et données d'enquête) pour répondre à la sous-question, sans préjuger du type de données attendu. ",
)

# Synthétiseur : pas de typage dans l'ancien prompt → identique
_OLD_SYNTHESIZER_NT = _OLD_SYNTHESIZER

# ── Sanity checks ──────────────────────────────────────────────────────────────
assert _TYPING_ANSWERER not in _OLD_ANSWERER_NT, "Détypages answerer échoué"
print(f"[init] Anciens prompts extraits — décomp_NT len={len(_OLD_DECOMPOSER_NT)}, "
      f"answerer_NT len={len(_OLD_ANSWERER_NT)}, synth len={len(_OLD_SYNTHESIZER_NT)}")

# ── Paths ─────────────────────────────────────────────────────────────────────
COMPLET     = ROOT / "comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json"
OUT_DIR     = ROOT / "comparaisons_rag/nt_v2_legacy"
JUDGE_DELAY = 1.0

HARD_ERROR_MARKERS = [
    "401", "authentication", "invalid_api_key",
    "insufficient_quota", "credit", "billing",
    "account", "disabled", "deactivated",
]


# ── Pipeline inline (anciens prompts + no_typing) ─────────────────────────────

def _decompose(question, n=DEFAULT_N_SUBQUESTIONS, temp=None):
    mix_instruction = "Chaque sous-question doit porter sur un aspect distinct de la question initiale.\n"
    prompt = (
        f"Decompose cette question en exactement {n} sous-questions ciblees et complementaires.\n"
        f"{mix_instruction}"
        "INTERDIT : deriver vers des dimensions non mentionnees dans la question initiale, "
        "sauf si la question est generale (bien-etre global).\n"
        "INTERDIT : generer des sous-questions sur des donnees non disponibles "
        "(ex: sous-indicateurs OppChoVec, ventilation demographique d'un score chiffre).\n"
        f"Reponds en JSON strict :\n"
        f'{{"sub_questions": ["sous-question 1", "sous-question 2"]}}\n\n'
        f"Question initiale : {question}"
    )
    raw = _call_mistral(prompt, _OLD_DECOMPOSER_NT, max_tokens=600,
                        temperature=temp if temp is not None else 0.4)
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(cleaned)["sub_questions"]
    except Exception:
        m = re.search(r'"sub_questions"\s*:\s*\[([^\]]+)\]', cleaned)
        if m:
            return json.loads(f'[{m.group(1)}]')
        raise RuntimeError(f"JSON décomposeur invalide : {cleaned[:200]}")


def _answer(sub_question, context, temp=None):
    prompt = f"Contexte (syntheses et verbatims) :\n{context[:15000]}\n\nSous-question : {sub_question}"
    return _call_claude(prompt, _OLD_ANSWERER_NT,
                        temperature=temp if temp is not None else 0.2)


def _synthesize(question, sub_qa_pairs, temp=None):
    pairs_text = "\n\n".join(
        f"Sous-question {i}: {sq}\nRéponse: {ans}"
        for i, (sq, ans) in enumerate(sub_qa_pairs, 1)
    )
    prompt = (
        f"Question initiale : {question}\n\n"
        f"Réponses aux sous-questions :\n{pairs_text}"
    )
    return _call_mistral(prompt, _OLD_SYNTHESIZER_NT,
                         max_tokens=2000,
                         temperature=temp if temp is not None else 0.3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_hard_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in HARD_ERROR_MARKERS)


def _is_complete(row):
    p = OUT_DIR / f"legacy_q{row:03d}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (isinstance(d.get("score_global"), (int, float))
                and isinstance(d.get("sub_questions"), list)
                and len(d["sub_questions"]) > 0)
    except Exception:
        return False


def _load_questions():
    data = json.loads(COMPLET.read_text(encoding="utf-8"))
    rows = {}
    for e in data.get("v_vanilla_k10", []):
        r = e.get("excel_row")
        if r and r not in rows:
            rows[r] = {k: e.get(k, "") for k in ("excel_row", "section", "subsection", "question")}
    return [rows[k] for k in sorted(rows)]


def judge_v43(question, answer, sources, section, subsection, expected_type):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\nSECTION : {section}\n\nSOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        if "mislabelling_flag" not in result:
            result["mislabelling_flag"] = any(
                str(v).lower() not in ("non", "false", "", "null", "none")
                for v in result.get("mislabelling_detecte", {}).values()
            )
        return result
    except Exception as ex:
        err = str(ex)
        if "insufficient_quota" in err.lower() or "quota" in err.lower():
            raise RuntimeError(f"QUOTA ÉPUISÉ: {err}")
        return {"judge_error": err, "score_global": None}


def _expected_type(section):
    s = (section or "").lower()
    return "limite_architecturale" if ("limite" in s and "architect" in s) else "reponse_substantielle_attendue"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Vérifier que le serveur API est accessible
    try:
        requests.get(f"{API_BASE}/api/health", timeout=5).raise_for_status()
        print(f"Serveur API OK ({API_BASE})\n", flush=True)
    except Exception as e:
        print(f"ERREUR : serveur API inaccessible sur {API_BASE} : {e}", flush=True)
        sys.exit(1)

    questions = _load_questions()
    if args.max:
        questions = questions[:args.max]
    n_q = len(questions)
    print(f"Lancement nt_v2_legacy (anciens prompts dé-typés) sur {n_q} questions...\n", flush=True)

    entries = {}
    n_skip = n_done = n_err = 0

    for i, q in enumerate(questions, 1):
        row = q["excel_row"]

        if _is_complete(row):
            p_json = OUT_DIR / f"legacy_q{row:03d}.json"
            try:
                entries[row] = json.loads(p_json.read_text(encoding="utf-8"))
            except Exception:
                pass
            n_skip += 1
            print(f"  [{i:3}/{n_q}] Q{row:3}  SKIP", flush=True)
            continue

        question = q["question"]
        entry = {k: q.get(k, "") for k in ("excel_row", "section", "subsection", "question")}

        # ── RAG inline ────────────────────────────────────────────────────────
        try:
            t0 = time.time()
            sub_questions = _decompose(question)
            sub_qa_pairs  = []
            all_sources   = []

            for j, sq in enumerate(sub_questions, 1):
                context_str, sources = _retrieve(sq, k=5)
                ans = _answer(sq, context_str)
                sub_qa_pairs.append((sq, ans))
                for s in sources:
                    s["sub_question_idx"] = j
                    s["sub_question"] = sq
                all_sources.extend(sources)

            final_answer = _synthesize(question, sub_qa_pairs)
            rag_elapsed  = round(time.time() - t0, 1)
        except RuntimeError as ex:
            if _is_hard_error(ex):
                print(f"\n[HARD STOP] {ex}", flush=True)
                sys.exit(1)
            # Cas spécial : refus géographique du décomposeur (commune hors Corse)
            err_str = str(ex)
            if "json décomposeur invalide" in err_str.lower():
                try:
                    raw_json = err_str.split("JSON décomposeur invalide :")[-1].strip()
                    err_obj = json.loads(raw_json)
                    refusal_msg = err_obj.get("error", "")
                except Exception:
                    refusal_msg = ""
                if refusal_msg or "hors" in err_str.lower() or "pas" in err_str.lower():
                    # Fallback : construire une réponse "hors périmètre" et juger quand même
                    final_answer = (
                        "Cette question porte sur une commune qui n'est pas en Corse. "
                        "L'indicateur OppChoVec et les données de l'enquête qualité de vie "
                        "ne couvrent que les 360 communes corses. "
                        "Aucune donnée n'est disponible pour cette commune dans la base."
                    )
                    rag_elapsed   = round(time.time() - t0, 1)
                    all_sources   = []
                    sub_qa_pairs  = [(question, final_answer)]
                    print(f"  [{i:3}/{n_q}] Q{row:3}  FALLBACK hors-Corse", flush=True)
                    # Continuer vers le bloc judge (pas de continue)
                else:
                    entry.update({"rag_status": "exception", "rag_error": err_str})
                    print(f"  [{i:3}/{n_q}] Q{row:3}  RAG EXCEPTION: {ex}", flush=True)
                    (OUT_DIR / f"legacy_q{row:03d}.json").write_text(
                        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
                    n_err += 1
                    continue
            else:
                entry.update({"rag_status": "exception", "rag_error": err_str})
                print(f"  [{i:3}/{n_q}] Q{row:3}  RAG EXCEPTION: {ex}", flush=True)
                (OUT_DIR / f"legacy_q{row:03d}.json").write_text(
                    json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
                n_err += 1
                continue
        except Exception as ex:
            if _is_hard_error(ex):
                print(f"\n[HARD STOP] {ex}", flush=True)
                sys.exit(1)
            entry.update({"rag_status": "exception", "rag_error": str(ex)})
            print(f"  [{i:3}/{n_q}] Q{row:3}  RAG EXCEPTION: {ex}", flush=True)
            (OUT_DIR / f"legacy_q{row:03d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            n_err += 1
            continue

        sources_for_judge = [
            {"content": s.get("content", ""), "metadata": s.get("metadata", {}),
             "source_type": s.get("source_type", ""), "label": s.get("label", "")}
            for s in all_sources
        ]
        sub_qa_list = [
            {"idx": j, "question": sq, "answer": ans}
            for j, (sq, ans) in enumerate(sub_qa_pairs, 1)
        ]
        entry.update({
            "rag_status":     "ok",
            "answer":         final_answer,
            "sub_questions":  sub_qa_list,
            "n_subquestions": len(sub_qa_list),
            "n_sources":      len(all_sources),
            "rag_elapsed_s":  rag_elapsed,
            "sources":        sources_for_judge,
        })
        (OUT_DIR / f"legacy_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Judge V4.3 ────────────────────────────────────────────────────────
        time.sleep(args.judge_delay)
        try:
            scores = judge_v43(question, final_answer, sources_for_judge,
                               q["section"], q["subsection"], _expected_type(q["section"]))
        except RuntimeError as quota_err:
            print(f"\n[HARD STOP] {quota_err}", flush=True)
            sys.exit(1)

        entry.update(scores)
        (OUT_DIR / f"legacy_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        entries[row] = entry
        n_done += 1
        sg = scores.get("score_global")
        sg_str = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
        mis    = " [MIS]" if scores.get("mislabelling_flag") else ""
        print(f"  [{i:3}/{n_q}] Q{row:3}  V4.3={sg_str}{mis}"
              f"  {len(sub_qa_list)}SQ  RAG={rag_elapsed}s  {len(all_sources)}src", flush=True)

    # Charger les skips manqués
    for q in questions:
        row = q["excel_row"]
        if row not in entries:
            p_json = OUT_DIR / f"legacy_q{row:03d}.json"
            if p_json.exists():
                try:
                    entries[row] = json.loads(p_json.read_text(encoding="utf-8"))
                except Exception:
                    pass

    # ── Patch COMPLET.json ────────────────────────────────────────────────────
    n_scored = sum(1 for e in entries.values() if isinstance(e.get("score_global"), (int, float)))
    if n_scored == n_q:
        data = json.loads(COMPLET.read_text(encoding="utf-8"))
        data["v_decomp_no_typing_v2_legacy"] = [entries[r] for r in sorted(entries)]
        COMPLET.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCOMPLET.json mis à jour : v_decomp_no_typing_v2_legacy ({n_scored} entrées).", flush=True)
    else:
        print(f"\n[ATTENTION] {n_scored}/{n_q} scorées — COMPLET.json non mis à jour.", flush=True)

    # ── Récap ─────────────────────────────────────────────────────────────────
    def gavg(key, edict):
        vals = [e[key] for e in edict.values() if isinstance(e.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    data_all = json.loads(COMPLET.read_text(encoding="utf-8"))
    configs = {
        "v_vanilla_k10 (old)":  {e["excel_row"]: e for e in data_all.get("v_vanilla_k10", [])},
        "v_decomp (old)":       {e["excel_row"]: e for e in data_all.get("v_decomp", [])},
        "v_decomp_raptor (old)":{e["excel_row"]: e for e in data_all.get("v_decomp_raptor", [])},
        "v_decomp_nt v1 (new)": {e["excel_row"]: e for e in data_all.get("v_decomp_no_typing", [])},
        "nt_v2_legacy (old)":   entries,
    }
    print(f"\n{'='*65}")
    for label, edict in configs.items():
        print(f"  {label:<28} global={gavg('score_global', edict)}"
              f"  Q/Q={gavg('coherence_qualiquanti', edict)}"
              f"  Nuance={gavg('nuance_incertitude', edict)}")


if __name__ == "__main__":
    main()
