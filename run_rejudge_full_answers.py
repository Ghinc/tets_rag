"""
run_rejudge_full_answers.py
Re-soumet au juge V4.3 les réponses stockées SANS troncature [:4000].
Cible : v_decomp et v_decomp_raptor, uniquement les entrées où len(answer) > 4000
(i.e. celles où l'ancien juge ne voyait qu'une fraction de la réponse).

Aucun appel Mistral / Anthropic : on utilise les réponses déjà stockées.
Seul coût : ~69 appels GPT-4o.
"""
import json, re, sys, time, copy
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

from eval_from_excel import (
    _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm
)

BACKUP_IN  = "comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET_backup_20260717_070832.json"
BACKUP_OUT = f"comparaisons_rag/ablations_103q_v43_gpt4o_REJUDGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

CONFIGS_TARGET = ["v_decomp", "v_decomp_raptor"]
JUDGE_DELAY    = 1.2  # s entre appels


def judge_full(question, answer, sources, section, subsection):
    """Juge V4.3 — réponse entière, sans [:4000]."""
    sources_text = _build_sources_text(sources)
    etype = "reponse_substantielle_attendue"
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {etype}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"   # ← PAS de [:4000]
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    try:
        raw    = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m      = re.search(r'\{[\s\S]*\}', raw)
        j      = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        return result
    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "insufficient_quota" in err:
            raise RuntimeError(f"QUOTA ÉPUISÉ: {err}")
        return {"judge_error": err, "score_global": None}


# ── Charger le backup
with open(BACKUP_IN, encoding="utf-8") as f:
    data = json.load(f)

# Deep copy pour ne pas modifier l'original
updated = copy.deepcopy(data)

SCORE_FIELDS = [
    "pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti",
    "score_global",
    "pertinence_justif", "fondement_factuel_justif",
    "nuance_incertitude_justif", "coherence_qualiquanti_justif",
    "raisonnement", "sources_inventaire", "mislabelling_detecte",
    "coherence_comportement_observe", "comportement_attendu_selon_grille",
    "judge_error", "judge_elapsed_s",
]

total_targets = 0
for cfg in CONFIGS_TARGET:
    total_targets += sum(1 for e in data.get(cfg, []) if len(e.get("answer", "")) > 4000)

print(f"\n{'='*70}")
print(f"RE-JUGEMENT RÉPONSES COMPLÈTES — {total_targets} entrées cibles")
print(f"Configs : {CONFIGS_TARGET}")
print(f"Critère : len(answer) > 4000 chars (juge ancien tronquait à 4000)")
print(f"{'='*70}\n")

done = 0
for cfg in CONFIGS_TARGET:
    entries   = data.get(cfg, [])
    u_entries = updated.get(cfg, [])
    targets   = [(i, e) for i, e in enumerate(entries) if len(e.get("answer", "")) > 4000]

    print(f"── {cfg} : {len(targets)} / {len(entries)} à re-juger")

    for idx, (i, e) in enumerate(targets, 1):
        q       = e["question"]
        answer  = e["answer"]
        sources = e.get("sources", [])
        section = e.get("section", "")
        sub     = e.get("subsection", "")
        old_sg  = e.get("score_global")
        judge_pct = f'{4000/len(answer)*100:.0f}%'

        print(f"  [{idx:2}/{len(targets)}] Q{e['excel_row']:3d}"
              f"  {len(answer):5d}ch  juge-ancien={judge_pct}"
              f"  old_sg={old_sg}"
              f"  {q[:50]}…", flush=True)

        time.sleep(JUDGE_DELAY)
        t0 = time.time()
        scores = judge_full(q, answer, sources, section, sub)
        elapsed = round(time.time() - t0, 1)

        new_sg = scores.get("score_global")
        delta  = round(float(new_sg) - float(old_sg), 2) if (new_sg and old_sg) else None
        sign   = f"{delta:+.2f}" if delta is not None else "?"

        if scores.get("judge_error"):
            print(f"         ❌ Erreur juge : {scores['judge_error'][:80]}", flush=True)
        else:
            print(f"         ✓ {elapsed}s  new_sg={new_sg}  [{sign}]", flush=True)

        # Mettre à jour l'entrée (scores + marqueur)
        for field in SCORE_FIELDS:
            if field in scores:
                u_entries[i][field] = scores[field]
        u_entries[i]["judge_elapsed_s"]   = elapsed
        u_entries[i]["rejudge_full"]      = True   # marqueur pour traçabilité
        u_entries[i]["score_global_orig"] = old_sg  # conserver l'ancien pour comparaison

        done += 1

# ── Sauvegarder
Path("comparaisons_rag").mkdir(exist_ok=True)
with open(BACKUP_OUT, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False, indent=2)

print(f"\n{'─'*70}")
print(f"TERMINÉ — {done} entrées re-jugées")
print(f"Nouveau backup : {BACKUP_OUT}")

# ── Résumé des deltas par config
print(f"\n{'─'*70}")
print(f"RÉSUMÉ DELTAS (nouveau score − ancien score)")
print(f"{'─'*70}")
for cfg in CONFIGS_TARGET:
    entries   = data.get(cfg, [])
    u_entries = updated.get(cfg, [])
    pairs = [
        (e.get("score_global"), u.get("score_global"))
        for e, u in zip(entries, u_entries)
        if u.get("rejudge_full") and e.get("score_global") and u.get("score_global")
    ]
    if not pairs: continue
    deltas = [float(n) - float(o) for o, n in pairs]
    mean_d = sum(deltas) / len(deltas)
    pos = sum(1 for d in deltas if d > 0.01)
    neg = sum(1 for d in deltas if d < -0.01)
    neu = len(deltas) - pos - neg
    c = '+' if mean_d > 0.01 else ('-' if mean_d < -0.01 else '=')
    print(f"  {cfg:<22}: Δ moyen = {c}{mean_d:+.3f}"
          f"  ↑{pos}  ↓{neg}  ={neu}  sur {len(deltas)} Q")
