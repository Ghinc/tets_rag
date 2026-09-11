"""
Test : re-générer 5 réponses avec max_tokens augmentés et vérifier qu'elles sont complètes.
Aucun appel juge — on vérifie uniquement la longueur et la fin de réponse.
"""
import json, re, sys, time, requests
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

BACKUP  = "comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET_backup_20260717_070832.json"
BASE    = "http://localhost:8000/api/query"

# 5 questions les plus flagrantes (mid-bold cutoff, v_decomp_raptor)
TARGET_ROWS = [93, 10, 71, 96, 36]
CFG         = "v_decomp_raptor"
K           = 5

with open(BACKUP, encoding="utf-8") as f:
    data = json.load(f)

old_by_row = {e["excel_row"]: e for e in data.get(CFG, [])}
q_info     = {e["excel_row"]: e for e in data.get("v_vanilla_k10", [])}

def ends_cleanly(ans):
    """True si la réponse se termine sur une fin de phrase/section."""
    s = ans.strip()
    return bool(re.search(r'[.!?»)\]]\*{0,2}\s*$|---\s*$|={3,}\s*$', s))

print(f"\n{'='*70}")
print(f"TEST SYNTHESIS — 5 questions, config={CFG}, max_tokens=6000 (nouveau)")
print(f"{'='*70}\n")

results = []
for row in TARGET_ROWS:
    q   = q_info[row]["question"]
    old = old_by_row.get(row, {})
    old_ans = old.get("answer", "")
    old_len = len(old_ans)
    old_end = repr(old_ans.strip()[-50:]) if old_ans else "?"

    print(f"Q{row:3d}  {q[:62]}…")
    print(f"  ANCIEN : {old_len}ch  fin={old_end}")

    t0 = time.time()
    try:
        resp = requests.post(BASE,
                             json={"question": q, "rag_version": CFG, "k": K},
                             headers={"Content-Type": "application/json"},
                             timeout=300)
        resp.raise_for_status()
        new_ans = resp.json().get("answer", "")
        elapsed = round(time.time() - t0, 1)
        new_len = len(new_ans)
        clean   = ends_cleanly(new_ans)
        new_end = repr(new_ans.strip()[-50:]) if new_ans else "?"

        status = "✓ COMPLET" if clean else "⚠ ENCORE TRONQUÉ"
        print(f"  NOUVEAU: {new_len}ch  {elapsed}s  {status}")
        print(f"  fin={new_end}")
        results.append({"row": row, "old_len": old_len, "new_len": new_len,
                        "complete": clean, "new_end": new_end})
    except Exception as e:
        print(f"  ERREUR: {e}")
        results.append({"row": row, "old_len": old_len, "new_len": None, "complete": False})
    print()

print(f"{'─'*70}")
print("BILAN:")
ok  = sum(1 for r in results if r["complete"])
nok = sum(1 for r in results if not r["complete"] and r["new_len"])
print(f"  ✓ Complètes  : {ok}/5")
print(f"  ⚠ Tronquées  : {nok}/5")
for r in results:
    if r["new_len"]:
        gain = r["new_len"] - r["old_len"]
        sign = f"+{gain}" if gain >= 0 else str(gain)
        icon = "✓" if r["complete"] else "⚠"
        print(f"  {icon} Q{r['row']:3d}  {r['old_len']}ch → {r['new_len']}ch [{sign}ch]")
