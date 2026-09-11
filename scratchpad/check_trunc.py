import json, re
from pathlib import Path

f = sorted(Path('comparaisons_rag').glob('ablations_103q_v43_gpt4o_RERUN_*.json'))[-1]
with open(f, encoding='utf-8') as fh:
    data = json.load(fh)

def sev(ans):
    s = ans.strip()
    if re.search(r'\*\*[A-Za-zÀ-ÿ0-9\s,]{0,40}$', s):
        if not re.search(r'\*\*[^*]+\*\*\s*$', s):
            return 3
    if not re.search(r'[.!?»)\]]\*{0,2}\s*$|---\s*$|={3,}\s*$', s):
        return 2
    return 0

print('Config                sev3  sev2  total')
for cfg in ['v_vanilla_k10','v_vanilla_k25','v_decomp','v_decomp_raptor']:
    entries = data.get(cfg, [])
    s3 = [e for e in entries if sev(e.get('answer',''))==3]
    s2 = [e for e in entries if sev(e.get('answer',''))==2]
    print(f'{cfg:<22}  {len(s3):3d}   {len(s2):3d}   {len(s3)+len(s2):3d}')

print()
print('Decomp+RAPTOR sev2 (re-run=True = déjà re-généré):')
for e in data.get('v_decomp_raptor', []):
    if sev(e.get('answer','')) == 2:
        ans = e.get('answer','').strip()
        rerun = e.get('rerun_synthesis', False)
        print(f"  Q{e['excel_row']:3d}  {len(ans)}ch  re-run={rerun}  fin={repr(ans[-50:])}")

print()
print('Decomp sev2 (re-run=True = déjà re-généré):')
for e in data.get('v_decomp', []):
    if sev(e.get('answer','')) == 2:
        ans = e.get('answer','').strip()
        rerun = e.get('rerun_synthesis', False)
        print(f"  Q{e['excel_row']:3d}  {len(ans)}ch  re-run={rerun}  fin={repr(ans[-50:])}")
