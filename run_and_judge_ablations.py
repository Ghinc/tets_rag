"""
run_and_judge_ablations.py — RAG query + LLM-judge en une seule passe.
Les sources complètes (2000 chars/source) sont passées directement au judge
sans passer par le JSON intermédiaire.

Usage:
    python run_and_judge_ablations.py --max 6       # test sur 6 questions
    python run_and_judge_ablations.py               # toutes les 20 questions
"""
import argparse, json, re, sys, time, requests, openpyxl
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_from_excel import (
    _call_llm, _build_sources_text,
    _JUDGE_V2_SYSTEM, _JUDGE_V41_SYSTEM, _parse_judge_v41,
)

XLSX    = r"C:\Users\comiti_g\Downloads\annotation_humaine_20q_final_v3_avec_juge.xlsx"
BASE    = "http://localhost:8000/api/query"
HEADERS = {"Content-Type": "application/json"}
VERSIONS = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
OUT_DIR  = "comparaisons_rag"

_JUDGE_PROMPT_V2 = (
    "QUESTION : {question}\n\nSECTION : {section}\n\n"
    "SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
    "RÉPONSE DU SYSTÈME :\n{answer}\n\n"
    "Évalue cette réponse. Suis la procédure en 5 étapes, résume-la en 1-2 phrases "
    "dans 'raisonnement', puis attribue les 4 notes. Pas de justification par dimension.\n\n"
    "Format JSON strict :\n{{\n"
    "  \"raisonnement\": \"<1-2 phrases, max 200 chars>\",\n"
    "  \"pertinence\": 1-5,\n  \"fondement_factuel\": 1-5,\n"
    "  \"nuance_incertitude\": 1-5,\n  \"coherence_qualiquanti\": 1-5,\n"
    "  \"applicable_sujet\": true|false,\n  \"note_sujet\": null|1-5,\n"
    "  \"sujet_evalue\": \"libellé court ou null\",\n"
    "  \"reason_non_applicable\": \"methodologique|refus|comparative|factuelle_brute|null\"\n}}"
)

_JUDGE_PROMPT_V41 = (
    "QUESTION : {question}\n\nSECTION : {section}\n\n"
    "TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
    "SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
    "RÉPONSE DU SYSTÈME :\n{answer}\n\n"
    "Évalue cette réponse selon la procédure et le format spécifiés."
)


def judge_v2(question: str, answer: str, sources: list, section: str) -> dict:
    sources_text = _build_sources_text(sources)
    prompt = _JUDGE_PROMPT_V2.format(
        question=question, section=section or "Non spécifiée",
        sources_text=sources_text, answer=answer[:4000],
    )
    try:
        raw = _call_llm(_JUDGE_V2_SYSTEM, prompt, max_tokens=600, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        notes = [j.get(k) for k in ("pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti")]
        return {
            "raisonnement":          j.get("raisonnement"),
            "pertinence":            j.get("pertinence"),
            "fondement_factuel":     j.get("fondement_factuel"),
            "nuance_incertitude":    j.get("nuance_incertitude"),
            "coherence_qualiquanti": j.get("coherence_qualiquanti"),
            "score_global":          round(sum(n for n in notes if n) / len([n for n in notes if n]), 2)
                                     if any(notes) else None,
            "applicable_sujet":      j.get("applicable_sujet"),
            "note_sujet":            j.get("note_sujet") if j.get("applicable_sujet") else None,
            "sujet_evalue":          j.get("sujet_evalue") if j.get("applicable_sujet") else None,
            "reason_non_applicable": j.get("reason_non_applicable"),
            "judge_error": None,
        }
    except Exception as e:
        return {"judge_error": str(e)}


def judge_v41(question: str, answer: str, sources: list, section: str,
              expected_type: str = "reponse_substantielle_attendue") -> dict:
    sources_text = _build_sources_text(sources)
    prompt = _JUDGE_PROMPT_V41.format(
        question=question, section=section or "Non spécifiée",
        expected_type=expected_type,
        sources_text=sources_text, answer=answer[:4000],
    )
    try:
        raw = _call_llm(_JUDGE_V41_SYSTEM, prompt, max_tokens=2000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        parsed = _parse_judge_v41(j)
        parsed["judge_error"] = None
        return parsed
    except Exception as e:
        return {"judge_error": str(e), "score_global": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=20, help="Nb de questions (défaut: 20)")
    parser.add_argument("--judge-delay", type=float, default=1.5, help="Délai entre appels judge (s)")
    parser.add_argument("--versions", nargs="+", default=None, help="Configs à tester (défaut: toutes)")
    args = parser.parse_args()

    wb = openpyxl.load_workbook(XLSX)
    ws = wb.active
    questions = []
    for r in range(2, ws.max_row + 1):
        questions.append({
            "excel_row": ws.cell(r, 1).value,
            "section":   ws.cell(r, 2).value,
            "subsection": ws.cell(r, 3).value,
            "question":  ws.cell(r, 4).value,
        })
    questions = questions[:args.max]
    versions = args.versions if args.versions else VERSIONS

    print(f"{len(questions)} questions x {len(versions)} configs = {len(questions)*len(versions)} appels RAG + judge", flush=True)

    results = {}
    for ver in versions:
        k = 10 if ver == "v_vanilla_k10" else (25 if ver == "v_vanilla_k25" else 5)
        print(f"\n{'='*58}", flush=True)
        print(f"Config: {ver}  (k={k})", flush=True)
        print('='*58, flush=True)
        results[ver] = []

        for i, q in enumerate(questions, 1):
            question = q["question"]
            entry = {"excel_row": q["excel_row"], "section": q["section"],
                     "subsection": q["subsection"], "question": question}

            # 1. Appel RAG
            try:
                t0 = time.time()
                r_resp = requests.post(BASE, json={"question": question, "rag_version": ver, "k": k},
                                       headers=HEADERS, timeout=300)
                rag_elapsed = time.time() - t0

                if r_resp.status_code != 200:
                    entry.update({"rag_status": "error", "rag_error": r_resp.text[:200]})
                    print(f"  [{i:2}/{len(questions)}] R{q['excel_row']} RAG ERREUR {r_resp.status_code}", flush=True)
                    results[ver].append(entry)
                    continue

                data = r_resp.json()
                raw_sources = data.get("sources", [])
                # Convertir au format attendu par _build_sources_text
                sources_for_judge = []
                for s in raw_sources:
                    sources_for_judge.append({
                        "content":   s.get("content") or s.get("extrait") or "",
                        "metadata":  s.get("metadata", {}),
                        "source_type": s.get("source_type", ""),
                        "label":     s.get("label", ""),
                    })

                entry.update({
                    "rag_status":     "ok",
                    "answer":         data.get("answer", ""),
                    "n_sources":      len(raw_sources),
                    "n_subquestions": len(data.get("sub_questions") or []),
                    "rag_elapsed_s":  round(rag_elapsed, 1),
                    "sources":        sources_for_judge,   # stocké pour les juges post-hoc
                })

            except Exception as e:
                entry.update({"rag_status": "exception", "rag_error": str(e)})
                print(f"  [{i:2}/{len(questions)}] R{q['excel_row']} RAG EXCEPTION: {e}", flush=True)
                results[ver].append(entry)
                continue

            # 2. Judge V2
            time.sleep(args.judge_delay)
            t0 = time.time()
            scores_v2 = judge_v2(question, entry["answer"], sources_for_judge, q["section"])
            judge_v2_elapsed = time.time() - t0
            entry.update(scores_v2)
            entry["judge_elapsed_s"] = round(judge_v2_elapsed, 1)

            # 3. Judge V4.1
            time.sleep(args.judge_delay)
            t0 = time.time()
            scores_v41 = judge_v41(question, entry["answer"], sources_for_judge, q["section"])
            judge_v41_elapsed = time.time() - t0
            for field, val in scores_v41.items():
                entry[f"v41_{field}"] = val
            entry["judge_v41_elapsed_s"] = round(judge_v41_elapsed, 1)

            sg     = scores_v2.get("score_global")
            sg41   = scores_v41.get("score_global")
            sg_str = f"{sg:.2f}" if sg else "?"
            sg41_s = f"{sg41:.2f}" if sg41 else "?"
            print(
                f"  [{i:2}/{len(questions)}] R{q['excel_row']} "
                f"V2={sg_str} V4.1={sg41_s} "
                f"| RAG {entry['rag_elapsed_s']}s | {entry['n_sources']}src",
                flush=True,
            )
            results[ver].append(entry)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_q = args.max
    out = Path(OUT_DIR) / f"ablations_{n_q}q_run_judge_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegarde : {out}", flush=True)

    # Tableau récap V2
    dims   = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti","score_global"]
    labels = ["Pertinence","Factuel","Nuance","Quali/Q","Global"]
    print(f"\n--- TABLEAU V2 (moyennes sur {n_q} questions) ---", flush=True)
    print(f"{'Config':<22} " + " ".join(f"{l:>10}" for l in labels), flush=True)
    print("-" * 80, flush=True)
    for ver in versions:
        ok = [e for e in results[ver] if e.get("rag_status") == "ok" and not e.get("judge_error")]
        def avg(key, entries=ok):
            vals = [entries[i][key] for i in range(len(entries)) if entries[i].get(key) is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "-"
        print(f"  {ver:<20} " + " ".join(f"{avg(d):>10}" for d in dims), flush=True)

    # Tableau récap V4.1
    dims41 = [f"v41_{d}" for d in dims[:-1]] + ["v41_score_global"]
    print(f"\n--- TABLEAU V4.1 (moyennes sur {n_q} questions) ---", flush=True)
    print(f"{'Config':<22} " + " ".join(f"{l:>10}" for l in labels), flush=True)
    print("-" * 80, flush=True)
    for ver in versions:
        ok41 = [e for e in results[ver] if e.get("rag_status") == "ok" and not e.get("v41_judge_error")]
        def avg41(key, entries=ok41):
            vals = [entries[i][key] for i in range(len(entries)) if entries[i].get(key) is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "-"
        print(f"  {ver:<20} " + " ".join(f"{avg41(d):>10}" for d in dims41), flush=True)


if __name__ == "__main__":
    main()
