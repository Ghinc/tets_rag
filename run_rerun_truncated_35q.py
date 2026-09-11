"""
run_rerun_truncated_35q.py
Re-génère + re-juge les réponses tronquées au niveau synthèse (layer 1).
Base : backup REJUDGE (scores juge [:4000] déjà corrigés).
Configs : v_decomp_raptor et v_decomp.
Critère : réponse se termine sans ponctuation finale propre (sev >= 2).
"""
import json, re, sys, time, copy, requests
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

BACKUP_IN  = "comparaisons_rag/ablations_103q_v43_gpt4o_REJUDGE_20260814_192707.json"
BACKUP_OUT = f"comparaisons_rag/ablations_103q_v43_gpt4o_RERUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

API_BASE   = "http://localhost:8000/api/query"
CONFIGS    = ["v_decomp_raptor", "v_decomp"]
K          = 5
JUDGE_DELAY = 1.2


def truncation_severity(ans):
    """
    3 = coupure flagrante : mid-bold-word (**Xxx sans fermeture)
    2 = mid-sentence : fin sans [.!?»)] et pas une liste/titre connu
    1 = suspicion légère
    0 = propre
    """
    s = ans.strip()
    if re.search(r'\*\*[A-Za-zÀ-ÿ0-9\s,]{0,40}$', s):
        # bold ouvert sans fermeture
        if not re.search(r'\*\*[^*]+\*\*\s*$', s):
            return 3
    if not re.search(r'[.!?»)\]]\*{0,2}\s*$|---\s*$|={3,}\s*$|:{0,1}\s*$', s):
        return 2
    return 0


def judge_full(question, answer, sources, section, subsection):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : reponse_substantielle_attendue\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"
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


# ── Charger backup
with open(BACKUP_IN, encoding="utf-8") as f:
    data = json.load(f)
updated = copy.deepcopy(data)

# ── Identifier les cibles (sev >= 2 dans chaque config)
targets_by_cfg = {}
for cfg in CONFIGS:
    targets = []
    for i, e in enumerate(data.get(cfg, [])):
        ans = e.get("answer", "")
        sev = truncation_severity(ans)
        if sev >= 2:
            targets.append((i, e, sev))
    targets_by_cfg[cfg] = targets

total = sum(len(t) for t in targets_by_cfg.values())
print(f"\n{'='*70}")
print(f"RE-RUN SYNTHÈSE TRONQUÉE — {total} entrées cibles")
for cfg in CONFIGS:
    print(f"  {cfg:<22}: {len(targets_by_cfg[cfg])} cibles")
print(f"{'='*70}\n")

SCORE_FIELDS = [
    "pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti",
    "score_global",
    "pertinence_justif", "fondement_factuel_justif",
    "nuance_incertitude_justif", "coherence_qualiquanti_justif",
    "raisonnement", "sources_inventaire", "mislabelling_detecte",
    "coherence_comportement_observe", "comportement_attendu_selon_grille",
    "judge_error", "judge_elapsed_s",
]

done = 0
errors = []

for cfg in CONFIGS:
    targets = targets_by_cfg[cfg]
    u_entries = updated.get(cfg, [])
    print(f"── {cfg} ({len(targets)} questions)")

    for idx, (i, e, sev) in enumerate(targets, 1):
        row = e["excel_row"]
        q   = e["question"]
        old_ans = e.get("answer", "")
        old_sg  = e.get("score_global")

        print(f"  [{idx:2}/{len(targets)}] Q{row:3d} sev={sev}  {len(old_ans)}ch"
              f"  sg={old_sg}  {q[:50]}…", flush=True)

        # ── 1. Re-générer via API
        for attempt in range(3):
            try:
                resp = requests.post(
                    API_BASE,
                    json={"question": q, "rag_version": cfg, "k": K},
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                resp.raise_for_status()
                new_ans = resp.json().get("answer", "")
                break
            except Exception as ex:
                if attempt == 2:
                    print(f"         ❌ API erreur ({ex})", flush=True)
                    errors.append({"cfg": cfg, "row": row, "step": "api", "err": str(ex)})
                    new_ans = None
                    break
                time.sleep(10)

        if not new_ans:
            continue

        new_sev = truncation_severity(new_ans)
        sev_icon = "✓" if new_sev == 0 else ("⚠" if new_sev == 2 else "❌")
        print(f"         {sev_icon} synthèse: {len(old_ans)}ch → {len(new_ans)}ch"
              f"  fin={repr(new_ans.strip()[-40:])}", flush=True)

        # ── 2. Re-juger
        time.sleep(JUDGE_DELAY)
        t0 = time.time()
        sources  = e.get("sources", [])
        section  = e.get("section", "")
        sub      = e.get("subsection", "")
        scores   = judge_full(q, new_ans, sources, section, sub)
        elapsed  = round(time.time() - t0, 1)
        new_sg   = scores.get("score_global")
        delta    = round(float(new_sg) - float(old_sg), 2) if (new_sg and old_sg) else None
        sign     = f"{delta:+.2f}" if delta is not None else "?"

        if scores.get("judge_error"):
            print(f"         ❌ Juge erreur: {scores['judge_error'][:80]}", flush=True)
            errors.append({"cfg": cfg, "row": row, "step": "judge", "err": scores["judge_error"]})
        else:
            print(f"         ✓ juge {elapsed}s  sg: {old_sg} → {new_sg}  [{sign}]", flush=True)

        # ── 3. Mettre à jour l'entrée
        u_entries[i]["answer"]           = new_ans
        u_entries[i]["score_global_orig_synthesis"] = old_sg  # avant ce rerun
        u_entries[i]["rerun_synthesis"]  = True
        u_entries[i]["judge_elapsed_s"]  = elapsed
        for field in SCORE_FIELDS:
            if field in scores:
                u_entries[i][field] = scores[field]

        done += 1
        print()

    print()

# ── Sauvegarder
Path("comparaisons_rag").mkdir(exist_ok=True)
with open(BACKUP_OUT, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False, indent=2)

print(f"{'─'*70}")
print(f"TERMINÉ — {done}/{total} entrées mises à jour")
print(f"Nouveau backup : {BACKUP_OUT}")
if errors:
    print(f"\n⚠ {len(errors)} erreurs :")
    for err in errors:
        print(f"  {err['cfg']} Q{err['row']} [{err['step']}] {err['err'][:60]}")

# ── Résumé des deltas par config
print(f"\n{'─'*70}")
print("RÉSUMÉ DELTAS (score_global)")
for cfg in CONFIGS:
    entries   = data.get(cfg, [])
    u_entries = updated.get(cfg, [])
    pairs = [
        (float(e.get("score_global", 0)), float(u.get("score_global", 0)))
        for e, u in zip(entries, u_entries)
        if u.get("rerun_synthesis") and e.get("score_global") and u.get("score_global")
    ]
    if not pairs: continue
    deltas = [n - o for o, n in pairs]
    mean_d = sum(deltas) / len(deltas)
    pos = sum(1 for d in deltas if d > 0.01)
    neg = sum(1 for d in deltas if d < -0.01)
    neu = len(deltas) - pos - neg
    sign = "+" if mean_d > 0.01 else ("-" if mean_d < -0.01 else "=")
    print(f"  {cfg:<22}: Δ moyen = {sign}{mean_d:+.3f}"
          f"  ↑{pos}  ↓{neg}  ={neu}  sur {len(pairs)} Q")
