"""
eval_from_excel.py — Pipeline d'évaluation automatisé du RAG v10

Lit les 103 questions de rag_evaluation_with_metrics_full.xlsx,
appelle l'API RAG pour chaque question, applique les métriques indiquées,
et exporte les résultats dans un rapport Markdown.

Usage:
    python eval_from_excel.py --max 5                          # test rapide
    python eval_from_excel.py --version v10 --output comparaisons_rag/
    python eval_from_excel.py --from-json results.json         # re-export uniquement
"""

import os
import sys
import json
import re
import time
import argparse
import traceback
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
EXCEL_INPUT = r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx"
API_BASE_URL = "http://localhost:8000"
RAG_VERSION = "v10"

JUDGE_MODEL = "mistral-large-latest"
JUDGE_BASE_URL = "https://api.mistral.ai/v1"
JUDGE_API_KEY_ENV = "MISTRAL_API_KEY"

# Noms des colonnes Excel → indices (1-based)
COL_SECTION     = 1  # A
COL_SUBSECTION  = 2  # B
COL_QUESTION    = 3  # C
COL_FACTUAL     = 4  # D
COL_BINARY      = 5  # E
COL_JUDGE       = 6  # F
COL_REFUSAL     = 7  # G
COL_HALLUC      = 8  # H
COL_OVERCONF    = 9  # I
COL_ROBUST      = 10 # J
COL_COMMENTS    = 11 # K


# ─────────────────────────────────────────────
# 1. Lecture de l'Excel
# ─────────────────────────────────────────────

def load_questions(excel_path: str) -> list[dict]:
    """Lit le fichier Excel et retourne une liste de dicts question+métriques."""
    import openpyxl
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    rows = []
    for r in range(2, ws.max_row + 1):
        q = ws.cell(r, COL_QUESTION).value
        if not q:
            continue
        rows.append({
            "excel_row": r,
            "section":    ws.cell(r, COL_SECTION).value or "",
            "subsection": ws.cell(r, COL_SUBSECTION).value or "",
            "question":   str(q).strip(),
            "do_factual":  ws.cell(r, COL_FACTUAL).value == "X",
            "do_binary":   ws.cell(r, COL_BINARY).value == "X",
            "do_judge":    ws.cell(r, COL_JUDGE).value == "X",
            "do_refusal":  ws.cell(r, COL_REFUSAL).value == "X",
            "do_halluc":   ws.cell(r, COL_HALLUC).value == "X",
            "do_overconf": ws.cell(r, COL_OVERCONF).value == "X",
            "do_robust":   ws.cell(r, COL_ROBUST).value == "X",
            "comments":    ws.cell(r, COL_COMMENTS).value or "",
        })
    return rows


# ─────────────────────────────────────────────
# 2. Appel API RAG (réutilisé de eval_multi_rag.py)
# ─────────────────────────────────────────────

def call_rag_api(question: str, rag_version: str = "v10", k: int = 7) -> dict:
    payload = {
        "question": question,
        "rag_version": rag_version,
        "k": k,
        "llm_model": "gpt-3.5-turbo",
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/api/query", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "answer": f"ERREUR API: {e}", "sources": []}


# ─────────────────────────────────────────────
# 3. Client Mistral (partagé)
# ─────────────────────────────────────────────

_mistral_client = None

def get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        from openai import OpenAI
        api_key = os.getenv(JUDGE_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Clé {JUDGE_API_KEY_ENV} non trouvée dans .env")
        _mistral_client = OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    return _mistral_client


def _call_llm(system: str, prompt: str, max_tokens: int = 800) -> str:
    """Appel LLM avec retry sur rate-limit."""
    client = get_mistral_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 2 ** attempt * 5
                print(f"    [RATE LIMIT] Attente {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ─────────────────────────────────────────────
# 4. Métriques
# ─────────────────────────────────────────────

def score_factual(question: str, answer: str, gt) -> dict:
    """
    Évalue la précision factuelle.
    - gt float → MAE + exact (tolérance 5 %)
    - gt str   → LLM vérifie si la réponse contient l'info attendue
    """
    if gt is None:
        return {"score": None, "detail": "Pas de ground truth disponible"}

    if isinstance(gt, (int, float)):
        # Extraction numérique
        numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", answer.replace(",", "."))
        candidates = []
        for n in numbers:
            try:
                candidates.append(float(n.replace(",", ".")))
            except ValueError:
                pass
        if not candidates:
            return {"score": 0.0, "detail": f"Aucun nombre trouvé dans la réponse (attendu {gt})"}
        # Prendre le nombre le plus proche
        best = min(candidates, key=lambda x: abs(x - gt))
        mae = abs(best - gt)
        exact = mae <= 0.05 * abs(gt) + 0.05  # tolérance relative 5% + absolue 0.05
        return {
            "score": 1.0 if exact else max(0.0, 1.0 - mae / (abs(gt) + 0.01)),
            "detail": f"Extrait={best:.2f}, Attendu={gt:.2f}, MAE={mae:.2f}, Exact={'oui' if exact else 'non'}",
        }

    else:  # gt est une chaîne — vérification LLM
        system = "Tu es un évaluateur factuel précis. Réponds UNIQUEMENT par un JSON valide."
        prompt = f"""Question : {question}
Réponse du système : {answer}
Valeur attendue : {gt}

La réponse contient-elle l'information attendue, explicitement ou de manière équivalente ?
Réponds en JSON :
{{"correct": true/false, "extrait": "la partie de la réponse qui correspond ou non", "explication": "..."}}"""
        try:
            raw = _call_llm(system, prompt, max_tokens=300)
            m = re.search(r'\{[\s\S]*\}', raw)
            j = json.loads(m.group()) if m else {}
            score = 1.0 if j.get("correct") else 0.0
            detail = j.get("explication", raw[:200])
            return {"score": score, "detail": detail}
        except Exception as e:
            return {"score": None, "detail": f"Erreur LLM: {e}"}


def score_binary(question: str, answer: str, expected: str) -> dict:
    """
    Extrait le label binaire de la réponse et le compare à `expected`.
    Retourne 1.0 si correct, 0.0 sinon.
    """
    if expected is None:
        # Pas de ground truth → évaluation par LLM seul (oui/non/incertain)
        system = "Tu es un évaluateur. Réponds UNIQUEMENT en JSON."
        prompt = f"""Question oui/non ou à choix binaire : {question}
Réponse : {answer}

La réponse fournit-elle une position claire ?
{{"position_claire": true/false, "label_extrait": "...", "confiance": "haute/moyenne/faible"}}"""
        try:
            raw = _call_llm(system, prompt, max_tokens=200)
            m = re.search(r'\{[\s\S]*\}', raw)
            j = json.loads(m.group()) if m else {}
            return {
                "score": None,
                "detail": f"Label={j.get('label_extrait','?')}, Clair={j.get('position_claire','?')}",
            }
        except Exception as e:
            return {"score": None, "detail": f"Erreur LLM: {e}"}

    system = "Tu es un évaluateur factuel. Réponds UNIQUEMENT en JSON."
    prompt = f"""Question : {question}
Réponse du système : {answer}
Label attendu : "{expected}"

Extrais le label binaire de la réponse et vérifie s'il correspond au label attendu.
Réponds en JSON :
{{"label_extrait": "...", "correct": true/false, "explication": "..."}}"""
    try:
        raw = _call_llm(system, prompt, max_tokens=300)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        score = 1.0 if j.get("correct") else 0.0
        detail = f"Extrait='{j.get('label_extrait','?')}', Attendu='{expected}' → {j.get('explication','')}"
        return {"score": score, "detail": detail}
    except Exception as e:
        return {"score": None, "detail": f"Erreur LLM: {e}"}


def score_judge(question: str, answer: str, sources: list, section: str) -> dict:
    """
    Juge LLM (Mistral Large) évaluant 4 dimensions (1-5).
    Adapté au corpus bien-être territorial Corse.
    """
    sources_text = ""
    for i, s in enumerate(sources, 1):
        content = s.get("content", s.get("extrait", ""))[:400]
        meta = {k: v for k, v in s.get("metadata", s).items()
                if k not in ("content", "extrait") and isinstance(v, (str, int, float))}
        meta_str = ", ".join(f"{k}={v}" for k, v in list(meta.items())[:4])
        sources_text += f"\n--- Source {i} [{meta_str}] ---\n{content}\n"

    system = """Tu es un évaluateur expert en analyse territoriale et bien-être en Corse.
Tu dois noter la qualité d'une réponse RAG sur 4 dimensions (chacune de 1 à 5).
Réponds UNIQUEMENT en JSON valide, sans texte avant ni après."""

    prompt = f"""=== QUESTION ===
{question}

=== SECTION ===
{section}

=== SOURCES RAG ===
{sources_text or "(aucune source fournie)"}

=== RÉPONSE ===
{answer}

Évalue et produis ce JSON :
{{
  "pertinence": NOTE_1_5,
  "fondement_factuel": NOTE_1_5,
  "nuance_incertitude": NOTE_1_5,
  "coherence_qualiquanti": NOTE_1_5,
  "score_global": MOYENNE_PONDEREE,
  "justification": "2-3 phrases résumant l'évaluation"
}}

Critères :
- pertinence (1-5) : la réponse traite bien la question posée
- fondement_factuel (1-5) : les affirmations s'appuient sur les sources
- nuance_incertitude (1-5) : le modèle exprime ses limites de façon appropriée
- coherence_qualiquanti (1-5) : équilibre quali/quanti adapté au type de question
- score_global : moyenne des 4 dimensions"""

    try:
        raw = _call_llm(system, prompt, max_tokens=500)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        # Recalcul du score global si absent ou incohérent
        dims = [j.get(k) for k in ("pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti")]
        dims_ok = [d for d in dims if isinstance(d, (int, float))]
        if dims_ok:
            j["score_global"] = round(sum(dims_ok) / len(dims_ok), 2)
        j["error"] = None
        return j
    except Exception as e:
        return {"error": str(e), "score_global": None}


def score_refusal_hallucination(question: str, answer: str) -> dict:
    """
    Pour les questions sur des données absentes (Nice, Haïti, 2018...).
    Détecte :
    - refusal_ok (bool) : le modèle refuse correctement de répondre
    - hallucination (bool) : le modèle invente une réponse factuelle
    - overconfidence (bool) : le modèle répond sans exprimer d'incertitude
    """
    # Heuristiques rapides avant d'appeler le LLM
    refusal_markers = [
        "n'ai pas", "ne dispose pas", "pas de données", "pas d'information",
        "hors de ma", "non disponible", "aucune donnée", "ne couvre pas",
        "je ne suis pas en mesure", "impossible de répondre",
        "données limitées", "pas dans les sources",
    ]
    lower = answer.lower()
    quick_refusal = any(m in lower for m in refusal_markers)

    system = "Tu es un évaluateur de systèmes RAG. Réponds UNIQUEMENT en JSON."
    prompt = f"""Question : {question}
Réponse du système : {answer}

Cette question porte sur des données inexistantes ou hors périmètre (ex : commune étrangère, année non couverte).
Le comportement attendu est de REFUSER de répondre, ou d'exprimer clairement l'absence de données.

Évalue :
{{
  "refusal_ok": true/false,  // le système refuse ou indique l'absence de données
  "hallucination": true/false,  // le système invente une réponse factuelle
  "overconfidence": true/false,  // le système répond sans exprimer d'incertitude
  "explication": "..."
}}"""
    try:
        raw = _call_llm(system, prompt, max_tokens=300)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        # Override si heuristique détecte un refus clair
        if quick_refusal and not j.get("refusal_ok"):
            j["refusal_ok"] = True
        return j
    except Exception as e:
        # Fallback sur heuristique
        return {
            "refusal_ok": quick_refusal,
            "hallucination": not quick_refusal,
            "overconfidence": not quick_refusal,
            "explication": f"Heuristique (LLM échoué: {e})",
        }


# ─────────────────────────────────────────────
# 5. Robustesse sémantique (post-traitement)
# ─────────────────────────────────────────────

def compute_semantic_robustness(groups: dict, row_answers: dict) -> dict:
    """
    Pour chaque groupe, embed les réponses et calcule la similarité cosinus intra-groupe.
    groups: {nom_groupe: [row1, row2, ...]}
    row_answers: {excel_row: answer_str}
    Retourne: {nom_groupe: {"mean_sim": float, "min_sim": float, "answers": [...]}}
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
        import numpy as np

        model_path = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
        if not os.path.exists(model_path):
            model_path = "BAAI/bge-m3"

        print("  Chargement BGE-M3 pour robustesse semantique...", flush=True)
        model = BGEM3FlagModel(model_path, use_fp16=True)

        results = {}
        for group_name, rows in groups.items():
            answers = [row_answers.get(r, "") for r in rows]
            non_empty = [a for a in answers if a and not a.startswith("ERREUR")]
            if len(non_empty) < 2:
                results[group_name] = {
                    "mean_sim": None, "min_sim": None,
                    "detail": "Moins de 2 réponses disponibles",
                    "answers": answers,
                }
                continue

            embeddings = model.encode(non_empty, batch_size=4, max_length=512)["dense_vecs"]
            # Cosinus pairwise
            sims = []
            n = len(embeddings)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = embeddings[i], embeddings[j]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                    sims.append(cos)

            results[group_name] = {
                "mean_sim": round(sum(sims) / len(sims), 4) if sims else None,
                "min_sim":  round(min(sims), 4) if sims else None,
                "detail":   f"{len(non_empty)} réponses, {len(sims)} paires",
                "answers":  answers,
            }
        return results

    except ImportError:
        print("  [AVERTISSEMENT] FlagEmbedding non disponible, robustesse ignorée", flush=True)
        return {g: {"mean_sim": None, "min_sim": None, "detail": "FlagEmbedding manquant", "answers": []}
                for g in groups}
    except Exception as e:
        traceback.print_exc()
        return {g: {"mean_sim": None, "min_sim": None, "detail": str(e), "answers": []}
                for g in groups}


# ─────────────────────────────────────────────
# 6. Export Excel (3 feuilles)
# ─────────────────────────────────────────────

def export_to_markdown(results: list[dict], robustness: dict, output_path: str,
                        metadata: dict = None):
    from collections import defaultdict

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else None
    def pct(v): return f"{v*100:.1f}%" if v is not None else "—"
    def fmt(v, suffix=""): return f"{v}{suffix}" if v is not None else "—"

    # ── Agrégation par section ──
    sec_data = defaultdict(lambda: {
        "n": 0, "factual_scores": [], "binary_scores": [],
        "judge_scores": [], "refusals": [], "hallucinations": [], "overconfs": [],
    })
    for r in results:
        sec = r.get("section", "Inconnue")
        d = sec_data[sec]
        d["n"] += 1
        f = r.get("scores", {})
        if f.get("factual", {}).get("score") is not None:
            d["factual_scores"].append(f["factual"]["score"])
        if f.get("binary", {}).get("score") is not None:
            d["binary_scores"].append(f["binary"]["score"])
        if f.get("judge", {}).get("score_global") is not None:
            d["judge_scores"].append(f["judge"]["score_global"])
        rs = f.get("refusal", {})
        if rs.get("refusal_ok") is not None:
            d["refusals"].append(1 if rs["refusal_ok"] else 0)
            d["hallucinations"].append(1 if rs.get("hallucination") else 0)
            d["overconfs"].append(1 if rs.get("overconfidence") else 0)

    md = []

    # ── En-tête ──
    ts = (metadata or {}).get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    version = (metadata or {}).get("rag_version", "?")
    judge_model = (metadata or {}).get("judge_model", "?")
    total_q = (metadata or {}).get("total_questions", len(results))

    md.append(f"# Rapport d'évaluation RAG — {version.upper()}\n")
    md.append(f"**Date :** {ts}  \n**Modèle juge :** {judge_model}  \n**Questions évaluées :** {total_q}\n")
    md.append("\n---\n")

    # ── Résumé global ──
    factual_ok = [r for r in results if r.get("scores", {}).get("factual", {}).get("score") is not None]
    binary_ok  = [r for r in results if r.get("scores", {}).get("binary",  {}).get("score") is not None]
    judge_ok   = [r for r in results if r.get("scores", {}).get("judge",   {}).get("score_global") is not None]
    refusal_ok = [r for r in results if r.get("scores", {}).get("refusal", {}).get("refusal_ok") is not None]
    errors     = [r for r in results if r.get("answer", "").startswith("ERREUR")]

    md.append("## Résumé global\n")
    md.append("| Métrique | Valeur | N |\n|---------|--------|---|\n")
    if factual_ok:
        md.append(f"| Factual accuracy | {pct(avg([r['scores']['factual']['score'] for r in factual_ok]))} | {len(factual_ok)} |\n")
    if binary_ok:
        md.append(f"| Binary accuracy | {pct(avg([r['scores']['binary']['score'] for r in binary_ok]))} | {len(binary_ok)} |\n")
    if judge_ok:
        score = avg([r['scores']['judge']['score_global'] for r in judge_ok])
        md.append(f"| Juge LLM (moy.) | {fmt(score, '/5')} | {len(judge_ok)} |\n")
    if refusal_ok:
        ref_rate = avg([1 if r['scores']['refusal']['refusal_ok'] else 0 for r in refusal_ok])
        hal_rate = avg([1 if r['scores']['refusal'].get('hallucination') else 0 for r in refusal_ok])
        md.append(f"| Refus correct | {pct(ref_rate)} | {len(refusal_ok)} |\n")
        md.append(f"| Hallucination | {pct(hal_rate)} | {len(refusal_ok)} |\n")
    if errors:
        md.append(f"| Erreurs API | {len(errors)} | {len(results)} |\n")
    md.append("\n")

    # ── Résumé par section ──
    md.append("## Résumé par section\n")
    md.append("| Section | N | Juge moy. | Binary acc. | Factual acc. | Refusal |\n")
    md.append("|---------|---|-----------|------------|--------------|--------|\n")
    for sec, d in sorted(sec_data.items()):
        md.append(
            f"| {sec} | {d['n']} "
            f"| {fmt(avg(d['judge_scores']), '/5')} "
            f"| {pct(avg(d['binary_scores']))} "
            f"| {pct(avg(d['factual_scores']))} "
            f"| {pct(avg(d['refusals']))} |\n"
        )
    md.append("\n")

    # ── Robustesse sémantique ──
    if robustness:
        md.append("## Robustesse sémantique\n")
        md.append("| Groupe | N | Sim. moy. | Sim. min. |\n|--------|---|-----------|----------|\n")
        for group, data in robustness.items():
            answers = data.get("answers", [])
            n_ok = len([a for a in answers if a and not a.startswith("ERREUR")])
            md.append(
                f"| {group} | {n_ok} "
                f"| {fmt(data.get('mean_sim'))} "
                f"| {fmt(data.get('min_sim'))} |\n"
            )
        md.append("\n> Interprétation : sim > 0.85 = très cohérent · 0.70–0.85 = acceptable · < 0.70 = fragile\n\n")

    # ── Résultats détaillés ──
    md.append("---\n\n## Résultats détaillés\n")

    current_section = None
    for r in results:
        sec = r.get("section", "")
        if sec != current_section:
            current_section = sec
            md.append(f"\n### {sec}\n")

        row = r.get("excel_row", "?")
        subsec = r.get("subsection", "")
        question = r.get("question", "")
        answer = r.get("answer", "")
        f = r.get("scores", {})
        metrics = r.get("metrics", [])

        md.append(f"\n#### R{row} — {subsec}\n\n")
        md.append(f"> {question}\n\n")

        # Badges métriques
        badge_map = {"factual": "📐 Factual", "binary": "🔢 Binary", "judge": "🧑‍⚖️ Juge",
                     "refusal": "🚫 Refusal", "halluc": "👻 Halluc.", "overconf": "⚠️ Overconf.", "robust": "🔄 Robustesse"}
        badges = " · ".join(badge_map.get(m, m) for m in metrics if m in badge_map)
        if badges:
            md.append(f"**Métriques :** {badges}\n\n")

        if answer.startswith("ERREUR"):
            md.append(f"❌ **{answer}**\n\n")
        else:
            md.append("**Réponse RAG :**\n\n")
            md.append(answer.strip() + "\n\n")

            # Scores
            scores_lines = []
            fs = f.get("factual", {})
            if fs.get("score") is not None:
                scores_lines.append(f"**📐 Factual :** {pct(fs['score'])} — {fs.get('detail', '')}")
            bs = f.get("binary", {})
            if bs.get("score") is not None:
                label = "✅" if bs["score"] == 1 else "❌"
                scores_lines.append(f"**🔢 Binary :** {label} {bs.get('detail', '')}")
            js = f.get("judge", {})
            if js.get("score_global") is not None:
                scores_lines.append(
                    f"**🧑‍⚖️ Juge :** {js['score_global']:.1f}/5 "
                    f"(pertinence: {fmt(js.get('pertinence'))} · "
                    f"fondement: {fmt(js.get('fondement_factuel'))} · "
                    f"nuance: {fmt(js.get('nuance_incertitude'))} · "
                    f"cohérence: {fmt(js.get('coherence_qualiquanti'))})"
                )
                if js.get("justification"):
                    scores_lines.append(f"*{js['justification'][:200]}{'…' if len(js.get('justification','')) > 200 else ''}*")
            rs = f.get("refusal", {})
            if rs.get("refusal_ok") is not None:
                ref_icon = "✅" if rs["refusal_ok"] else "❌"
                hal_icon = "⚠️" if rs.get("hallucination") else "✅"
                oc_icon  = "⚠️" if rs.get("overconfidence") else "✅"
                scores_lines.append(
                    f"**🚫 Refusal :** {ref_icon} · "
                    f"Hallucination : {hal_icon} · "
                    f"Overconfidence : {oc_icon}"
                )
                if rs.get("explication"):
                    scores_lines.append(f"*{rs['explication'][:200]}*")
            rb = r.get("robustness_group", "")
            if rb:
                scores_lines.append(f"**🔄 Groupe robustesse :** `{rb}`")
            if r.get("comments"):
                scores_lines.append(f"*{r['comments']}*")

            if scores_lines:
                md.append("\n".join(scores_lines) + "\n\n")

        md.append("---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"Markdown sauvegarde : {output_path}")


# ─────────────────────────────────────────────
# 7. Pipeline principal
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluation RAG depuis Excel — pipeline complet")
    parser.add_argument("--input",   default=EXCEL_INPUT, help="Fichier Excel d'entrée")
    parser.add_argument("--version", default=RAG_VERSION, help="Version RAG à tester (default: v10)")
    parser.add_argument("--k",       type=int, default=7, help="Nombre de chunks à récupérer")
    parser.add_argument("--max",     type=int, default=0, help="Limiter à N questions (0=toutes)")
    parser.add_argument("--rows",    default="", help="Lignes Excel à évaluer, ex: '2,5,7-12' (vide=toutes)")
    parser.add_argument("--output",  default="comparaisons_rag", help="Dossier de sortie")
    parser.add_argument("--no-judge",   action="store_true", help="Désactiver le juge LLM")
    parser.add_argument("--no-robust",  action="store_true", help="Désactiver la robustesse sémantique")
    parser.add_argument("--from-json",  default=None, help="Re-exporter depuis un JSON existant")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Mode re-export ──
    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        results = saved["results"]
        robustness = saved.get("robustness", {})
        md_path = args.from_json.replace(".json", "_reexport.md")
        export_to_markdown(results, robustness, md_path, metadata=saved.get("metadata", {}))
        return

    # ── Chargement ground truth ──
    from eval_ground_truth import get_full_ground_truth, BINARY_EXPECTED, ROBUSTNESS_GROUPS, find_ground_truth
    gt = get_full_ground_truth()
    print(f"Ground truth charge : {len(gt)} entrees")

    # ── Lecture Excel ──
    questions = load_questions(args.input)

    # Filtrage par numéros de lignes (--rows "2,5,7-12")
    if args.rows.strip():
        selected_rows = set()
        for part in args.rows.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                selected_rows.update(range(int(a), int(b) + 1))
            elif part.isdigit():
                selected_rows.add(int(part))
        questions = [q for q in questions if q["excel_row"] in selected_rows]

    if args.max > 0:
        questions = questions[:args.max]
    total = len(questions)

    print("=" * 70)
    print(f"EVALUATION RAG — {total} questions — version {args.version}")
    print(f"Juge LLM : {JUDGE_MODEL}")
    print("=" * 70)

    results = []
    row_answers = {}  # excel_row → answer (pour robustesse)

    for i, q in enumerate(questions, 1):
        question   = q["question"]
        excel_row  = q["excel_row"]
        section    = q["section"]

        print(f"\n[{i}/{total}] R{excel_row} {question[:65]}...", flush=True)

        # ── 1. Appel RAG ──
        print("  RAG...", end=" ", flush=True)
        api_resp = call_rag_api(question, args.version, k=args.k)
        if "error" in api_resp and api_resp.get("answer", "").startswith("ERREUR"):
            print(f"ERREUR: {api_resp['error']}")
            results.append({**q, "answer": api_resp["answer"], "sources": [], "scores": {}})
            row_answers[excel_row] = api_resp["answer"]
            continue

        answer  = api_resp.get("answer", "")
        sources = api_resp.get("sources", [])
        print(f"OK ({len(sources)} sources)", flush=True)
        row_answers[excel_row] = answer

        scores = {}

        # ── 2. Factual Accuracy ──
        if q["do_factual"]:
            gt_val, found = find_ground_truth(question, gt)
            comment = "" if found else "(pas de ground truth — skippé)"
            print(f"  Factual {'[GT='+str(gt_val)+']' if found else '[GT manquant]'}...",
                  end=" ", flush=True)
            if found:
                scores["factual"] = score_factual(question, answer, gt_val)
                print(f"score={scores['factual'].get('score')}", flush=True)
            else:
                scores["factual"] = {"score": None, "detail": "Ground truth non disponible"}
                q["comments"] = (q.get("comments") or "") + " " + comment
                print("skip", flush=True)

        # ── 3. Binary ──
        if q["do_binary"]:
            expected = BINARY_EXPECTED.get(question)
            print(f"  Binary [attendu={expected}]...", end=" ", flush=True)
            scores["binary"] = score_binary(question, answer, expected)
            print(f"score={scores['binary'].get('score')}", flush=True)

        # ── 4. LLM-as-a-Judge ──
        if q["do_judge"] and not args.no_judge:
            print("  Judge LLM...", end=" ", flush=True)
            scores["judge"] = score_judge(question, answer, sources, section)
            sg = scores["judge"].get("score_global")
            print(f"global={sg}/5" if sg else f"erreur={scores['judge'].get('error')}", flush=True)
        elif q["do_judge"]:
            scores["judge"] = {"score_global": None, "detail": "Judge desactive"}

        # ── 5. Refusal / Hallucination / Overconfidence ──
        if q["do_refusal"] or q["do_halluc"] or q["do_overconf"]:
            print("  Refusal/Halluc/Overconf...", end=" ", flush=True)
            scores["refusal"] = score_refusal_hallucination(question, answer)
            r_ok = scores["refusal"].get("refusal_ok")
            hall = scores["refusal"].get("hallucination")
            print(f"refusal={r_ok}, halluc={hall}", flush=True)

        # ── Robustesse : pas encore calculée (post-traitement) ──
        rob_group = ""
        for gname, rows in ROBUSTNESS_GROUPS.items():
            if excel_row in rows:
                rob_group = gname
                break

        results.append({
            **q,
            "answer":          answer,
            "sources":         sources,
            "scores":          scores,
            "robustness_group": rob_group,
        })

        # Pause anti-rate-limit
        time.sleep(1)

    # ── 6. Robustesse sémantique ──
    robustness = {}
    if not args.no_robust:
        print("\n[Robustesse semantique]")
        robustness = compute_semantic_robustness(ROBUSTNESS_GROUPS, row_answers)
        for g, d in robustness.items():
            sim = d.get("mean_sim")
            print(f"  {g}: sim_moy={sim}")

    # ── 7. Export ──
    base = f"eval_from_excel_{args.version}_{timestamp}"

    json_path = os.path.join(args.output, f"{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Rendre les sources sérialisables
        for r in results:
            r["sources"] = [
                {k: v for k, v in s.items() if isinstance(v, (str, int, float, bool, type(None)))}
                if isinstance(s, dict) else str(s)
                for s in r.get("sources", [])
            ]
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "rag_version": args.version,
                "judge_model": JUDGE_MODEL,
                "total_questions": total,
            },
            "results": results,
            "robustness": robustness,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nJSON sauvegarde : {json_path}")

    md_path = os.path.join(args.output, f"{base}.md")
    export_to_markdown(results, robustness, md_path, metadata={
        "timestamp": timestamp,
        "rag_version": args.version,
        "judge_model": JUDGE_MODEL,
        "total_questions": total,
    })

    # ── 8. Résumé terminal ──
    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)
    factual_ok  = [r for r in results if r.get("scores", {}).get("factual", {}).get("score") is not None]
    binary_ok   = [r for r in results if r.get("scores", {}).get("binary",  {}).get("score") is not None]
    judge_ok    = [r for r in results if r.get("scores", {}).get("judge",   {}).get("score_global") is not None]
    refusal_ok  = [r for r in results if r.get("scores", {}).get("refusal", {}).get("refusal_ok") is not None]

    def safe_avg(lst, key_fn): return round(sum(key_fn(r) for r in lst) / len(lst), 3) if lst else "N/A"

    print(f"  Factual   : {len(factual_ok)} questions, score moyen = {safe_avg(factual_ok, lambda r: r['scores']['factual']['score'])}")
    print(f"  Binary    : {len(binary_ok)} questions, accuracy = {safe_avg(binary_ok, lambda r: r['scores']['binary']['score'])}")
    print(f"  Judge     : {len(judge_ok)} questions, score moyen = {safe_avg(judge_ok, lambda r: r['scores']['judge']['score_global'])}/5")
    refusal_rate = safe_avg(refusal_ok, lambda r: 1 if r['scores']['refusal']['refusal_ok'] else 0)
    halluc_rate  = safe_avg(refusal_ok, lambda r: 1 if r['scores']['refusal'].get('hallucination') else 0)
    print(f"  Refusal   : {len(refusal_ok)} questions, taux refus correct = {refusal_rate}, hallucination = {halluc_rate}")
    for g, d in robustness.items():
        print(f"  Robust [{g[:25]}] : sim_moy={d.get('mean_sim')}, sim_min={d.get('min_sim')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
