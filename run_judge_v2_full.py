"""
run_judge_v2_full.py

Re-juge les 103 questions avec le prompt V2 (GPT-4o few-shot) en réutilisant
les réponses RAG et sources déjà présentes dans le JSON de la dernière éval.
Ne rappelle pas l'API RAG.

Sorties (dans comparaisons_rag/) :
  - eval_from_excel_v10_judge_v2_<ts>.json   : JSON complet avec scores V1+V2
  - judge_v1_vs_v2_report_<ts>.md            : comparatif V1 / V2

Usage :
    python run_judge_v2_full.py
    python run_judge_v2_full.py --input comparaisons_rag/eval_from_excel_v10_20260429_230109.json
"""

import argparse
import json
import logging
import os
import sys
import io
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("comparaisons_rag")

# JSON le plus récent avec 103 questions
DEFAULT_INPUT = str(OUTPUT_DIR / "eval_from_excel_v10_20260429_230109.json")


def find_latest_full_eval() -> str:
    """Retourne le chemin du JSON le plus récent avec 103 questions."""
    candidates = sorted(OUTPUT_DIR.glob("eval_from_excel_v10_*.json"), reverse=True)
    for p in candidates:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if len(d.get("results", [])) >= 100:
                return str(p)
        except Exception:
            continue
    raise FileNotFoundError("Aucun JSON eval_from_excel avec ≥100 questions trouvé.")


def sources_from_result(result: dict) -> list:
    """
    Normalise les sources du JSON eval vers le format attendu par score_judge :
    [{"content": "...", "metadata": {...}}, ...]
    """
    raw = result.get("sources") or []
    out = []
    for s in raw:
        if isinstance(s, dict):
            out.append({
                "content":  s.get("content") or s.get("extrait") or "",
                "metadata": s.get("metadata") or {},
            })
    return out


def run_rejudge(input_path: str, resume_from: str | None = None) -> tuple[list, dict]:
    log.info("Lecture : %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results_in = data.get("results", [])
    total = len(results_in)
    log.info("Questions à re-juger : %d", total)

    # Reprise : charger les résultats déjà traités depuis un JSON partiel
    already_done: dict[int, dict] = {}
    if resume_from and Path(resume_from).exists():
        with open(resume_from, encoding="utf-8") as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            jv2 = (r.get("scores") or {}).get("judge_v2") or {}
            if not jv2.get("error") and jv2.get("score_global") is not None:
                already_done[r.get("excel_row")] = r
        log.info("Reprise : %d questions déjà traitées chargées", len(already_done))

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_from_excel import score_judge

    results_out = []
    for i, r in enumerate(results_in, 1):
        row_id   = r.get("excel_row")
        question = r.get("question", "")
        answer   = r.get("answer", "")
        section  = r.get("section", "")

        # Réutiliser si déjà traité avec succès
        if row_id in already_done:
            log.info("[%d/%d] R%s : reprise (skip)", i, total, row_id)
            results_out.append(already_done[row_id])
            continue

        sources = sources_from_result(r)
        log.info("[%d/%d] R%s : %s", i, total, row_id, question[:70])

        j_v2 = score_judge(question, answer, sources, section)

        if j_v2.get("error"):
            log.warning("   ÉCHEC : %s", j_v2["error"])
        else:
            sg_v1 = (r.get("scores") or {}).get("judge", {}).get("score_global")
            log.info("   V1=%s  V2=%s", sg_v1, j_v2.get("score_global"))

        entry = deepcopy(r)
        entry.setdefault("scores", {})
        entry["scores"]["judge_v1"] = entry["scores"].pop("judge", None)
        entry["scores"]["judge_v2"] = j_v2
        results_out.append(entry)
        time.sleep(1)

    meta = {
        "judge_v2_model":   "gpt-4o",
        "judge_v2_prompt":  "V2-few-shot",
        "judge_v1_source":  str(input_path),
        "total_questions":  total,
        "timestamp":        datetime.now().isoformat(),
    }
    return results_out, meta


def compute_agreement(results: list) -> dict:
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        log.warning("scipy/numpy non disponibles — métriques ignorées")
        return {}

    dims = ("pertinence", "fondement_factuel", "nuance_incertitude",
            "coherence_qualiquanti", "score_global")
    versions = ("judge_v1", "judge_v2")
    metrics = {v: {} for v in versions}

    # Pour accord inter-versions (V1 vs V2)
    v1v2_pairs = {d: ([], []) for d in dims}

    for r in results:
        scores = r.get("scores", {})
        for d in dims:
            v1 = (scores.get("judge_v1") or {}).get(d)
            v2 = scores.get("judge_v2", {}).get(d)
            if v1 is not None and v2 is not None and not scores.get("judge_v2", {}).get("error"):
                try:
                    v1v2_pairs[d][0].append(float(v1))
                    v1v2_pairs[d][1].append(float(v2))
                except (TypeError, ValueError):
                    pass

    for d in dims:
        v1_arr = np.array(v1v2_pairs[d][0])
        v2_arr = np.array(v1v2_pairs[d][1])
        if len(v1_arr) < 3:
            continue
        pr, pp = stats.pearsonr(v1_arr, v2_arr)
        sr, _  = stats.spearmanr(v1_arr, v2_arr)
        metrics["inter_version"][d if d != "score_global" else "score_global"] = {
            "n":          len(v1_arr),
            "pearson_r":  round(float(pr), 3),
            "pearson_p":  round(float(pp), 3),
            "spearman":   round(float(sr), 3),
            "mae":        round(float(np.mean(np.abs(v1_arr - v2_arr))), 3),
            "bias_v1_minus_v2": round(float(np.mean(v1_arr - v2_arr)), 3),
            "v2_mean":    round(float(np.mean(v2_arr)), 3),
            "v1_mean":    round(float(np.mean(v1_arr)), 3),
        }

    return metrics


def compute_agreement(results: list) -> dict:
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        log.warning("scipy/numpy non disponibles — métriques ignorées")
        return {}

    dims = ("pertinence", "fondement_factuel", "nuance_incertitude",
            "coherence_qualiquanti", "score_global")

    pairs = {d: ([], []) for d in dims}
    for r in results:
        scores = r.get("scores", {})
        jv1 = scores.get("judge_v1") or {}
        jv2 = scores.get("judge_v2") or {}
        if jv2.get("error"):
            continue
        for d in dims:
            v1 = jv1.get(d)
            v2 = jv2.get(d)
            if v1 is not None and v2 is not None:
                try:
                    pairs[d][0].append(float(v1))
                    pairs[d][1].append(float(v2))
                except (TypeError, ValueError):
                    pass

    metrics = {}
    for d in dims:
        a1 = np.array(pairs[d][0])
        a2 = np.array(pairs[d][1])
        if len(a1) < 3:
            continue
        pr, pp = stats.pearsonr(a1, a2)
        sr, _  = stats.spearmanr(a1, a2)
        metrics[d] = {
            "n":               len(a1),
            "v1_mean":         round(float(np.mean(a1)), 3),
            "v2_mean":         round(float(np.mean(a2)), 3),
            "bias_v1_minus_v2": round(float(np.mean(a1 - a2)), 3),
            "mae":             round(float(np.mean(np.abs(a1 - a2))), 3),
            "pearson_r_v1v2":  round(float(pr), 3),
            "pearson_p_v1v2":  round(float(pp), 3),
            "spearman_v1v2":   round(float(sr), 3),
        }
    return metrics


def build_report(results: list, metrics: dict, ts: str, input_path: str) -> str:
    n_ok  = sum(1 for r in results if not (r.get("scores", {}).get("judge_v2") or {}).get("error"))
    n_err = len(results) - n_ok

    lines = [
        "# Comparatif Judge V1 (zero-shot) vs V2 (few-shot) — 103 questions",
        "",
        f"**Date** : {ts}  ",
        f"**Modèle** : gpt-4o  ",
        f"**Source V1** : `{Path(input_path).name}`  ",
        f"**Questions traitées** : {n_ok}/{len(results)}"
        + (f"  (**{n_err} erreurs**)" if n_err else ""),
        "",
        "## Accord inter-version V1 ↔ V2",
        "",
        "| Dimension | n | Moy V1 | Moy V2 | Biais V1−V2 | MAE | Pearson | p |",
        "|-----------|---|--------|--------|-------------|-----|---------|---|",
    ]
    for d, m in metrics.items():
        lines.append(
            f"| {d} | {m['n']} | {m['v1_mean']} | {m['v2_mean']} "
            f"| {m['bias_v1_minus_v2']} | {m['mae']} "
            f"| {m['pearson_r_v1v2']} | {m['pearson_p_v1v2']} |"
        )

    lines += [
        "",
        "## Distribution des écarts V1 − V2 (score_global)",
        "",
    ]
    sg_m = metrics.get("score_global")
    if sg_m:
        lines += [
            f"- Moyenne V1 : **{sg_m['v1_mean']}**",
            f"- Moyenne V2 : **{sg_m['v2_mean']}**",
            f"- Biais (V1−V2) : **{sg_m['bias_v1_minus_v2']}** "
            f"({'V1 plus sévère' if sg_m['bias_v1_minus_v2'] > 0 else 'V2 plus sévère'})",
            f"- MAE inter-version : **{sg_m['mae']}**",
            f"- Pearson V1↔V2 : **{sg_m['pearson_r_v1v2']}** (p={sg_m['pearson_p_v1v2']})",
        ]

    lines += ["", "## Détail par question (score_global)", "",
              "| Row | Question | V1 | V2 | Δ |",
              "|-----|----------|----|----|---|"]
    for r in results:
        jv1 = (r.get("scores") or {}).get("judge_v1") or {}
        jv2 = (r.get("scores") or {}).get("judge_v2") or {}
        sg1 = jv1.get("score_global")
        sg2 = jv2.get("score_global")
        if jv2.get("error"):
            lines.append(f"| R{r.get('excel_row','?')} | {r.get('question','')[:60]} | {sg1} | ERR | — |")
        else:
            delta = round(float(sg1) - float(sg2), 2) if sg1 is not None and sg2 is not None else "—"
            arrow = ("↑" if isinstance(delta, float) and delta > 0.1
                     else "↓" if isinstance(delta, float) and delta < -0.1 else "≈")
            lines.append(f"| R{r.get('excel_row','?')} | {r.get('question','')[:60]} | {sg1} | {sg2} | {delta} {arrow} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="JSON eval source (défaut : dernier eval_from_excel_v10_*.json)")
    parser.add_argument("--resume", default=None,
                        help="JSON partiel précédent à reprendre (skip les rows déjà OK)")
    args = parser.parse_args()

    input_path = args.input or find_latest_full_eval()
    log.info("Source : %s", input_path)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results, meta = run_rejudge(input_path, resume_from=args.resume)

    # Sauvegarde JSON complet
    out_data = {"meta": meta, "results": results}
    json_path = OUTPUT_DIR / f"eval_from_excel_v10_judge_v2_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    log.info("JSON sauvegardé : %s", json_path)

    # Métriques + rapport
    metrics = compute_agreement(results)
    ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = build_report(results, metrics, ts_readable, input_path)
    md_path = OUTPUT_DIR / f"judge_v1_vs_v2_report_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    log.info("Rapport comparatif : %s", md_path)

    # Résumé console
    sg = metrics.get("score_global", {})
    if sg:
        print("\n" + "=" * 60)
        print("COMPARATIF V1 vs V2 — score_global (103 questions)")
        print(f"  Moy. V1          : {sg['v1_mean']}")
        print(f"  Moy. V2          : {sg['v2_mean']}")
        print(f"  Biais V1−V2      : {sg['bias_v1_minus_v2']}")
        print(f"  MAE inter-version: {sg['mae']}")
        print(f"  Pearson V1↔V2   : {sg['pearson_r_v1v2']}  (p={sg['pearson_p_v1v2']})")
        print("=" * 60)


if __name__ == "__main__":
    main()
