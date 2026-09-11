"""
run_rerun_all_remaining.py
Second pass : re-génère + re-juge TOUT ce qui reste tronqué (sev>=2).
Base : backup RERUN le plus récent (sev=3 decomp/DR déjà corrigés).
Cibles : decomp/DR sev=2 manqués + vanilla sev>=2 (max_tokens=4000 désormais).
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

# ── Fichiers
BASE_DIR  = Path("comparaisons_rag")
rerun_files = sorted(BASE_DIR.glob("ablations_103q_v43_gpt4o_RERUN_*.json"))
if not rerun_files:
    sys.exit("Aucun fichier RERUN trouvé.")
BACKUP_IN  = rerun_files[-1]
BACKUP_OUT = BASE_DIR / f"ablations_103q_v43_gpt4o_RERUN2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

API_BASE    = "http://localhost:8000/api/query"
CONFIGS     = ["v_decomp_raptor", "v_decomp", "v_vanilla_k10", "v_vanilla_k25"]
CFG_K       = {"v_decomp_raptor": 5, "v_decomp": 5, "v_vanilla_k10": 10, "v_vanilla_k25": 25}
JUDGE_DELAY = 1.2


def trunc_sev(ans):
    """Détection sans bug :{0,1}."""
    s = ans.strip()
    if re.search(r'\*\*[A-Za-zÀ-ÿ0-9\s,]{0,40}$', s):
        if not re.search(r'\*\*[^*]+\*\*\s*$', s):
            return 3
    if not re.search(r'[.!?»)\]]\*{0,2}\s*$|---\s*$|={3,}\s*$', s):
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


# ── Charger
print(f"Base : {BACKUP_IN.name}")
with open(BACKUP_IN, encoding="utf-8") as f:
    data = json.load(f)
updated = copy.deepcopy(data)

# ── Identifier les cibles (sev>=2)
targets_by_cfg = {}
for cfg in CONFIGS:
    targets = [
        (i, e)
        for i, e in enumerate(data.get(cfg, []))
        if trunc_sev(e.get("answer", "")) >= 2
    ]
    targets_by_cfg[cfg] = targets

total = sum(len(t) for t in targets_by_cfg.values())
print(f"\n{'='*70}")
print(f"RE-RUN COMPLET — {total} entrées cibles (sev>=2)")
for cfg in CONFIGS:
    n = len(targets_by_cfg[cfg])
    print(f"  {cfg:<22}: {n}")
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
    targets   = targets_by_cfg[cfg]
    u_entries = updated.get(cfg, [])
    k         = CFG_K[cfg]
    print(f"── {cfg} k={k} ({len(targets)} questions)")

    for idx, (i, e) in enumerate(targets, 1):
        row     = e["excel_row"]
        q       = e["question"]
        old_ans = e.get("answer", "")
        old_sg  = e.get("score_global")
        old_sev = trunc_sev(old_ans)

        print(f"  [{idx:2}/{len(targets)}] Q{row:3d} sev={old_sev}  {len(old_ans)}ch"
              f"  sg={old_sg}  {q[:48]}…", flush=True)

        # ── 1. Re-générer
        new_ans = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    API_BASE,
                    json={"question": q, "rag_version": cfg, "k": k},
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                resp.raise_for_status()
                new_ans = resp.json().get("answer", "")
                break
            except Exception as ex:
                if attempt == 2:
                    print(f"         ❌ API ({attempt+1}/3) {ex}", flush=True)
                    errors.append({"cfg": cfg, "row": row, "step": "api", "err": str(ex)})
                else:
                    time.sleep(15 * (attempt + 1))

        if not new_ans:
            continue

        new_sev  = trunc_sev(new_ans)
        sev_icon = "✓" if new_sev == 0 else ("⚠" if new_sev == 2 else "❌")
        print(f"         {sev_icon} {len(old_ans)}ch→{len(new_ans)}ch"
              f"  sev:{old_sev}→{new_sev}"
              f"  fin={repr(new_ans.strip()[-45:])}", flush=True)

        # ── 2. Re-juger
        time.sleep(JUDGE_DELAY)
        t0      = time.time()
        scores  = judge_full(q, new_ans, e.get("sources", []),
                             e.get("section", ""), e.get("subsection", ""))
        elapsed = round(time.time() - t0, 1)
        new_sg  = scores.get("score_global")
        delta   = round(float(new_sg) - float(old_sg), 2) if (new_sg and old_sg) else None
        sign    = f"{delta:+.2f}" if delta is not None else "?"

        if scores.get("judge_error"):
            print(f"         ❌ Juge : {scores['judge_error'][:80]}", flush=True)
            errors.append({"cfg": cfg, "row": row, "step": "judge", "err": scores["judge_error"]})
        else:
            print(f"         ✓ juge {elapsed}s  sg: {old_sg}→{new_sg}  [{sign}]", flush=True)

        # ── 3. Mettre à jour
        u_entries[i]["answer"]                     = new_ans
        u_entries[i]["rerun_synthesis2"]           = True
        u_entries[i]["score_global_orig_rerun2"]   = old_sg
        u_entries[i]["judge_elapsed_s"]            = elapsed
        for field in SCORE_FIELDS:
            if field in scores:
                u_entries[i][field] = scores[field]

        done += 1
        print()

# ── Sauvegarder
with open(BACKUP_OUT, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False, indent=2)

print(f"{'─'*70}")
print(f"TERMINÉ — {done}/{total} mises à jour  |  {len(errors)} erreurs")
print(f"Nouveau backup : {BACKUP_OUT}")

# ── Check final troncature
print(f"\n{'─'*70}")
print("CHECK TRONCATURE FINAL")
for cfg in CONFIGS:
    entries = updated.get(cfg, [])
    s3 = sum(1 for e in entries if trunc_sev(e.get("answer",""))==3)
    s2 = sum(1 for e in entries if trunc_sev(e.get("answer",""))==2)
    icon = "✓" if s3==0 and s2==0 else "⚠"
    print(f"  {icon} {cfg:<22}: sev3={s3}  sev2={s2}")

# ── Deltas globaux
print(f"\n{'─'*70}")
print("DELTAS SCORE_GLOBAL (vs backup base)")
with open(BACKUP_IN, encoding="utf-8") as f:
    base = json.load(f)

def mean(vals):
    v = [float(x) for x in vals if x is not None]
    return round(sum(v)/len(v), 3) if v else None

for cfg in CONFIGS:
    old_scores = [e.get("score_global") for e in base.get(cfg,[])]
    new_scores = [e.get("score_global") for e in updated.get(cfg,[])]
    om = mean(old_scores); nm = mean(new_scores)
    if om and nm:
        d = round(nm-om, 3)
        sign = "+" if d >= 0 else ""
        print(f"  {cfg:<22}: {om:.3f} → {nm:.3f}  [{sign}{d:.3f}]")
