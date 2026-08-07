"""
run_no_typing_109q.py — Run complet v_decomp_no_typing × 109 questions + juge V4.3 (GPT-4o).

Checkpointing PAR QUESTION (JSONL append) : relancer la même commande reprend
exactement là où le run s'est interrompu, sans rien refaire.

Usage :
    python run_no_typing_109q.py           # premier lancement OU reprise
    python run_no_typing_109q.py --max 5   # test rapide sur 5 questions

Gestion des erreurs :
    TRANSITOIRE (timeout, 5xx, 429) → retry avec backoff exponentiel (4 essais max)
    DURE (401/402/403, quota, billing, crédits) → arrêt immédiat, checkpoint écrit,
      message clair + commande de reprise affichée
"""

import argparse, json, re, sys, time, requests, openpyxl
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

# ── Forcer GPT-4o comme juge ───────────────────────────────────────────────
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

# ── Constantes ──────────────────────────────────────────────────────────────
XLSX        = r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx"
API_BASE    = "http://localhost:8000/api/query"
HEADERS     = {"Content-Type": "application/json"}
RAG_VERSION = "v_decomp_no_typing"
K           = 5
OUT_DIR     = Path("comparaisons_rag")
CKPT_JSONL  = OUT_DIR / "no_typing_checkpoint.jsonl"
FINAL_JSON  = OUT_DIR / "no_typing_109q_FINAL.json"
COMPLET_JSON = OUT_DIR / "ablations_103q_v43_gpt4o_COMPLET.json"
JUDGE_DELAY = 1.0
PROGRESS_EVERY = 10   # log toutes les N questions

# Mots-clés → erreur DURE (arrêt immédiat)
_HARD_KEYWORDS = [
    "insufficient_quota", "insufficient funds", "insufficient_funds",
    "quota", "billing", "payment", "credits exhausted",
    "unauthorized", "api key", "invalid_api_key",
    "rate limit exceeded permanently",
]
_HARD_STATUS = {401, 402, 403}


# ── Erreurs ─────────────────────────────────────────────────────────────────

class HardError(RuntimeError):
    """Erreur non-transitoire → arrêt immédiat."""
    pass


def _is_hard(text: str, status: int | None = None) -> bool:
    if status in _HARD_STATUS:
        return True
    low = text.lower()
    return any(kw in low for kw in _HARD_KEYWORDS)


# ── Retry wrapper ────────────────────────────────────────────────────────────

def _with_retry(fn, *args, max_tries=4, context="", **kwargs):
    """Appelle fn(*args, **kwargs) avec retry backoff. Lève HardError si erreur dure."""
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except HardError:
            raise
        except Exception as exc:
            msg = str(exc)
            if _is_hard(msg):
                raise HardError(msg)
            if attempt < max_tries - 1:
                wait = 5 * (2 ** attempt)   # 5, 10, 20, 40s
                print(f"    [retry {attempt+1}/{max_tries}] {context}: {msg[:120]} — attente {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                raise   # transitoire persistant → on lève quand même


# ── Chargement questions ─────────────────────────────────────────────────────

def load_questions(max_q=None):
    wb = openpyxl.load_workbook(XLSX)
    ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        section    = (ws.cell(r, 1).value or "").strip()
        subsection = (ws.cell(r, 2).value or "").strip()
        question   = (ws.cell(r, 3).value or "").strip()
        if not question:
            continue
        rows.append({
            "excel_row":  r - 1,
            "section":    section,
            "subsection": subsection,
            "question":   question,
        })
    return rows[:max_q] if max_q else rows


def _expected_type(section: str) -> str:
    s = section.lower()
    if "limite" in s and "architect" in s:
        return "limite_architecturale"
    return "reponse_substantielle_attendue"


# ── Checkpoint JSONL ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Retourne {excel_row: entry} pour les questions complètes (RAG ok + juge ok)."""
    done = {}
    if not CKPT_JSONL.exists():
        return done
    with open(CKPT_JSONL, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] JSONL ligne {lineno} corrompue — ignorée", flush=True)
                continue
            row = entry.get("excel_row")
            if row is None:
                continue
            # On garde cette entrée si RAG ok ET pas d'erreur juge
            if entry.get("rag_status") == "ok" and not entry.get("judge_error"):
                done[row] = entry
    return done


def append_checkpoint(entry: dict):
    """Ajoute une entrée au JSONL (append atomique)."""
    OUT_DIR.mkdir(exist_ok=True)
    with open(CKPT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Appel RAG ─────────────────────────────────────────────────────────────────

def call_rag(question: str) -> dict:
    def _do():
        resp = requests.post(
            API_BASE,
            json={"question": question, "rag_version": RAG_VERSION, "k": K},
            headers=HEADERS,
            timeout=300,
        )
        if resp.status_code in _HARD_STATUS:
            raise HardError(f"HTTP {resp.status_code} — {resp.text[:200]}")
        if resp.status_code != 200:
            body = resp.text[:300]
            if _is_hard(body, resp.status_code):
                raise HardError(f"HTTP {resp.status_code} — {body}")
            raise RuntimeError(f"HTTP {resp.status_code} — {body}")
        data = resp.json()
        # Détecter quota Mistral/Anthropic dans la réponse
        answer = data.get("answer", "")
        if _is_hard(answer):
            raise HardError(f"Erreur dure dans la réponse RAG : {answer[:200]}")
        return data
    return _with_retry(_do, context="RAG")


# ── Appel juge ─────────────────────────────────────────────────────────────────

def call_judge(question: str, answer: str, sources: list,
               section: str, subsection: str, expected_type: str) -> dict:
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )

    def _do():
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        if _is_hard(raw):
            raise HardError(f"Erreur dure dans réponse juge : {raw[:200]}")
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

    try:
        return _with_retry(_do, context="judge")
    except HardError:
        raise
    except Exception as exc:
        err = str(exc)
        if _is_hard(err):
            raise HardError(err)
        return {"judge_error": err, "score_global": None}


# ── Sauvegarde finale ─────────────────────────────────────────────────────────

def save_final(entries: list) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    payload = {RAG_VERSION: entries}
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  [FINAL] {FINAL_JSON}  ({len(entries)} entrées)", flush=True)
    return FINAL_JSON


def merge_into_complet(entries: list):
    """Fusionne v_decomp_no_typing dans ablations_103q_v43_gpt4o_COMPLET.json."""
    if not COMPLET_JSON.exists():
        print(f"  [WARN] {COMPLET_JSON} introuvable — fusion ignorée", flush=True)
        return
    with open(COMPLET_JSON, encoding="utf-8") as f:
        complet = json.load(f)
    complet[RAG_VERSION] = entries
    # Backup avant écrasement
    backup = COMPLET_JSON.with_name(COMPLET_JSON.stem + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import shutil
    shutil.copy2(COMPLET_JSON, backup)
    with open(COMPLET_JSON, "w", encoding="utf-8") as f:
        json.dump(complet, f, ensure_ascii=False, indent=2)
    print(f"  [COMPLET] {COMPLET_JSON} mis à jour (backup → {backup.name})", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Limiter à N questions (test)")
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    questions  = load_questions(args.max)
    n_q        = len(questions)
    start_time = datetime.now()

    # ── Reprise : charger ce qui est déjà fait ────────────────────────────────
    done = load_checkpoint()
    n_done_at_start = len(done)
    if n_done_at_start:
        print(f"[REPRISE] {CKPT_JSONL.name} — {n_done_at_start} questions déjà complètes, reprend ici.",
              flush=True)
    else:
        print(f"[NOUVEAU RUN] Checkpoint : {CKPT_JSONL}", flush=True)

    print(f"\n{n_q} questions × {RAG_VERSION} (k={K}) + juge GPT-4o/V4.3", flush=True)
    print(f"À faire : {n_q - n_done_at_start} questions restantes", flush=True)
    print(f"{'='*65}\n", flush=True)

    # ── Boucle principale ─────────────────────────────────────────────────────
    n_done_this_run = 0
    try:
        for i, q in enumerate(questions, 1):
            row = q["excel_row"]

            # Déjà complet → skip
            if row in done:
                print(f"  [{i:3}/{n_q}] Q{row:3}  SKIP (déjà complet)", flush=True)
                continue

            question  = q["question"]
            etype     = _expected_type(q["section"])
            entry     = {
                "excel_row":  row,
                "section":    q["section"],
                "subsection": q["subsection"],
                "question":   question,
            }

            # ── 1. Appel RAG ───────────────────────────────────────────────
            try:
                t0 = time.time()
                data = call_rag(question)
                rag_elapsed = round(time.time() - t0, 1)

                raw_sources = data.get("sources", [])
                sources_for_judge = [
                    {
                        "content":    s.get("content") or s.get("extrait") or "",
                        "metadata":   s.get("metadata", {}),
                        "source_type": s.get("source_type", ""),
                        "label":      s.get("label", ""),
                    }
                    for s in raw_sources
                ]
                entry.update({
                    "rag_status":     "ok",
                    "answer":         data.get("answer", ""),
                    "n_sources":      len(raw_sources),
                    "n_subquestions": len(data.get("sub_questions") or []),
                    "rag_elapsed_s":  rag_elapsed,
                    "sources":        sources_for_judge,
                })

            except HardError as exc:
                _hard_stop(exc, row, n_done_at_start + n_done_this_run, n_q,
                           done, entry, "RAG")
                return  # never reached — _hard_stop calls sys.exit

            except Exception as exc:
                print(f"  [{i:3}/{n_q}] Q{row:3}  RAG ERREUR (transitoire épuisée): {exc}", flush=True)
                entry.update({"rag_status": "error", "rag_error": str(exc)[:300]})
                # On n'écrit PAS au JSONL (pas complet) — mais on continue
                continue

            # ── 2. Juge V4.3 ──────────────────────────────────────────────
            time.sleep(args.judge_delay)
            try:
                t0 = time.time()
                scores = call_judge(
                    question, entry["answer"], sources_for_judge,
                    q["section"], q["subsection"], etype,
                )
                judge_elapsed = round(time.time() - t0, 1)
                entry.update(scores)
                entry["judge_elapsed_s"] = judge_elapsed

            except HardError as exc:
                _hard_stop(exc, row, n_done_at_start + n_done_this_run, n_q,
                           done, entry, "juge GPT-4o")
                return

            except Exception as exc:
                print(f"  [{i:3}/{n_q}] Q{row:3}  JUGE ERREUR (transitoire épuisée): {exc}", flush=True)
                entry.update({"judge_error": str(exc)[:300], "score_global": None})
                # Pas d'entrée dans done → sera rejugé au prochain run

            # ── 3. Écriture checkpoint ─────────────────────────────────────
            # N'écrire que si RAG ok ET juge ok (pas d'erreur juge)
            if entry.get("rag_status") == "ok" and not entry.get("judge_error"):
                append_checkpoint(entry)
                done[row] = entry
                n_done_this_run += 1

            sg     = entry.get("score_global")
            sg_str = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
            mis    = " [MIS]" if entry.get("mislabelling_flag") else ""
            print(
                f"  [{i:3}/{n_q}] Q{row:3}  V4.3={sg_str}{mis}"
                f"  RAG={entry.get('rag_elapsed_s','?')}s"
                f"  juge={entry.get('judge_elapsed_s','?')}s"
                f"  {entry.get('n_sources','?')}src",
                flush=True,
            )

            # ── 4. Log de progression toutes les PROGRESS_EVERY questions ──
            n_total_done = n_done_at_start + n_done_this_run
            if n_done_this_run > 0 and n_done_this_run % PROGRESS_EVERY == 0:
                elapsed = datetime.now() - start_time
                rate    = n_done_this_run / elapsed.total_seconds() * 60
                eta_min = (n_q - n_total_done) / rate if rate > 0 else 0
                print(
                    f"\n  *** PROGRESSION : {n_total_done}/{n_q} complètes"
                    f"  |  ce run : {n_done_this_run}  |  écoulé : {_fmt_elapsed(elapsed)}"
                    f"  |  ETA : ~{eta_min:.0f} min ***\n",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(f"\n[INTERROMPU] Ctrl+C — {len(done)} questions sauvées dans {CKPT_JSONL}", flush=True)
        print(f"Relancer avec : python run_no_typing_109q.py", flush=True)
        sys.exit(0)

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    all_entries = list(done.values())
    all_entries.sort(key=lambda e: e.get("excel_row", 0))

    print(f"\n{'='*65}", flush=True)
    print(f"RUN TERMINÉ — {len(all_entries)}/{n_q} entrées complètes — {_fmt_elapsed(elapsed)}", flush=True)

    dims   = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti","score_global"]
    labels = ["Pertinence","Factuel","Nuance","Quali/Q","Global"]
    ok = [e for e in all_entries if e.get("rag_status") == "ok" and not e.get("judge_error")]
    print(f"\n{'─'*65}", flush=True)
    print(f"{RAG_VERSION} — {len(ok)} questions jugées avec succès")
    print("  " + "  ".join(f"{l:>10}" for l in labels))

    def avg(key):
        vals = [e[key] for e in ok if isinstance(e.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.2f}" if vals else "-"

    print("  " + "  ".join(f"{avg(d):>10}" for d in dims))
    n_mis = sum(1 for e in ok if e.get("mislabelling_flag"))
    print(f"  Mislabelling : {n_mis}/{len(ok)}")

    # Sauvegarde finale + fusion COMPLET
    save_final(all_entries)
    merge_into_complet(all_entries)

    # Questions manquantes
    done_rows = {e["excel_row"] for e in all_entries}
    all_rows  = {q["excel_row"] for q in questions}
    missing   = sorted(all_rows - done_rows)
    if missing:
        print(f"\n  [WARN] {len(missing)} questions sans résultat complet : {missing}", flush=True)
        print(f"  Relancer : python run_no_typing_109q.py", flush=True)
    else:
        print(f"\nToutes les questions sont complètes. ✓", flush=True)


def _fmt_elapsed(td: timedelta) -> str:
    total_s = int(td.total_seconds())
    h, rem  = divmod(total_s, 3600)
    m, s    = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _hard_stop(exc: HardError, row: int, n_done: int, n_total: int,
               done: dict, entry: dict, provider: str):
    """Arrêt propre sur erreur dure. Affiche diagnostic + commande de reprise."""
    print(f"\n{'!'*65}", flush=True)
    print(f"ERREUR DURE — {provider}", flush=True)
    print(f"Message : {exc}", flush=True)
    print(f"Question sur laquelle ça s'est arrêté : Q{row}", flush=True)
    print(f"Progression : {n_done}/{n_total} questions complètes dans {CKPT_JSONL}", flush=True)
    print(f"", flush=True)
    print(f"Action requise : vérifiez les crédits / clé API du fournisseur '{provider}'", flush=True)
    print(f"Commande pour reprendre : python run_no_typing_109q.py", flush=True)
    print(f"{'!'*65}", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
