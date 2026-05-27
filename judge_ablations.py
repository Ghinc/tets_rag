"""
judge_ablations.py — LLM-judge V2 (scores uniquement, sans justifications)
pour l'étude d'ablation sur les 20 questions annotées.

Usage:
    python judge_ablations.py --input comparaisons_rag/ablations_20q_YYYYMMDD_HHMMSS.json
"""
import argparse, json, re, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_from_excel import _call_llm, _build_sources_text, _JUDGE_V2_SYSTEM

# Format de sortie allégé : notes uniquement, pas de justifications
_USER_PROMPT_TEMPLATE = (
    "QUESTION : {question}\n\n"
    "SECTION : {section}\n\n"
    "SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
    "RÉPONSE DU SYSTÈME :\n{answer}\n\n"
    "Évalue cette réponse. Suis la procédure en 5 étapes, résume-la en 1-2 phrases dans 'raisonnement', "
    "puis attribue les 4 notes. Pas de justification par dimension.\n\n"
    "Format JSON strict :\n"
    "{{\n"
    "  \"raisonnement\": \"<1-2 phrases résumant l'évaluation, max 200 chars>\",\n"
    "  \"pertinence\": 1-5,\n"
    "  \"fondement_factuel\": 1-5,\n"
    "  \"nuance_incertitude\": 1-5,\n"
    "  \"coherence_qualiquanti\": 1-5,\n"
    "  \"applicable_sujet\": true|false,\n"
    "  \"note_sujet\": null|1-5,\n"
    "  \"sujet_evalue\": \"libellé court ou null\",\n"
    "  \"reason_non_applicable\": \"methodologique|refus|comparative|factuelle_brute|null\"\n"
    "}}"
)


def _build_sources_short(sources: list, max_sources: int = 40, max_chars: int = 300) -> str:
    """Version allégée : 40 sources × 300 chars pour couvrir les réponses riches."""
    text = ""
    for i, s in enumerate(sources[:max_sources], 1):
        content = (s.get("content") or s.get("extrait") or "")[:max_chars]
        label = s.get("label") or s.get("source_type") or ""
        text += f"\n--- Source {i} [{label}] ---\n{content}\n"
    return text or "(aucune source fournie)"


def judge_one(question: str, answer: str, sources: list, section: str) -> dict:
    sources_text = _build_sources_short(sources)
    prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        section=section or "Non spécifiée",
        sources_text=sources_text,
        answer=answer[:3000],
    )
    try:
        raw = _call_llm(_JUDGE_V2_SYSTEM, prompt, max_tokens=600, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        return {
            "pertinence":           j.get("pertinence"),
            "fondement_factuel":    j.get("fondement_factuel"),
            "nuance_incertitude":   j.get("nuance_incertitude"),
            "coherence_qualiquanti": j.get("coherence_qualiquanti"),
            "score_global":         round(
                sum(filter(None, [
                    j.get("pertinence"), j.get("fondement_factuel"),
                    j.get("nuance_incertitude"), j.get("coherence_qualiquanti"),
                ])) / 4, 2
            ) if all(j.get(k) for k in ("pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti")) else None,
            "applicable_sujet":     j.get("applicable_sujet"),
            "note_sujet":           j.get("note_sujet") if j.get("applicable_sujet") else None,
            "sujet_evalue":         j.get("sujet_evalue") if j.get("applicable_sujet") else None,
            "reason_non_applicable": j.get("reason_non_applicable"),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON ablations_20q_*.json")
    parser.add_argument("--delay", type=float, default=2.0, help="Délai entre appels (s)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    versions = list(data.keys())
    print(f"Configs: {versions}")
    print(f"Questions par config: {len(next(iter(data.values())))}")
    print(f"Total appels GPT-4o: {sum(len(v) for v in data.values())}\n")

    results = {ver: [] for ver in versions}

    for ver in versions:
        print(f"\n{'='*55}")
        print(f"Config: {ver}")
        print('='*55)
        entries = data[ver]
        for i, entry in enumerate(entries, 1):
            if entry.get("status") != "ok":
                print(f"  [{i:2}/20] R{entry.get('excel_row')} SKIP (status={entry.get('status')})")
                results[ver].append({"excel_row": entry.get("excel_row"), "skipped": True})
                continue

            question = entry["question"]
            answer = entry.get("answer", "")
            section = entry.get("section", "")
            # Les sources ne sont pas dans le JSON ablations (on n'a que n_sources)
            # On passe une liste vide — le judge évalue sur question+réponse uniquement
            sources = []

            t0 = time.time()
            scores = judge_one(question, answer, sources, section)
            elapsed = time.time() - t0

            scores["excel_row"] = entry.get("excel_row")
            scores["question"] = question
            scores["section"] = section
            scores["config"] = ver
            results[ver].append(scores)

            if scores.get("error"):
                print(f"  [{i:2}/20] R{entry.get('excel_row')} ERREUR: {scores['error'][:80]}")
            else:
                sg = scores.get("score_global")
                print(f"  [{i:2}/20] R{entry.get('excel_row')} "
                      f"P={scores.get('pertinence')} F={scores.get('fondement_factuel')} "
                      f"N={scores.get('nuance_incertitude')} C={scores.get('coherence_qualiquanti')} "
                      f"-> {sg:.2f} | {elapsed:.1f}s", flush=True)

            time.sleep(args.delay)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.input).parent
    out_path = out_dir / f"ablations_20q_judge_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nScores sauvegardés : {out_path}")

    # Tableau récap
    print("\n--- TABLEAU RÉCAPITULATIF (moyennes sur 20 questions) ---")
    print(f"{'Config':<22} {'Pertinence':>10} {'Factuel':>8} {'Nuance':>7} {'Quali/Q':>8} {'Global':>7}")
    print("-" * 65)
    for ver in versions:
        ok = [e for e in results[ver] if not e.get("skipped") and not e.get("error")]
        def avg(key):
            vals = [e[key] for e in ok if e.get(key) is not None]
            return round(sum(vals)/len(vals), 2) if vals else None
        p, f, n, c, g = avg("pertinence"), avg("fondement_factuel"), avg("nuance_incertitude"), avg("coherence_qualiquanti"), avg("score_global")
        print(f"  {ver:<20} {p or '-':>10} {f or '-':>8} {n or '-':>7} {c or '-':>8} {g or '-':>7}")


if __name__ == "__main__":
    main()
