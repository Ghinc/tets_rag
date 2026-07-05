"""Compare decomp+raptor avec et sans bilan booléen sur 5 questions."""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from rag_v10_raptor_subq import RaptorSubQuestionPipeline

TEST_ROWS = [82, 15, 4, 95, 36]

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    data = json.load(f)
questions = {e['excel_row']: e['question'] for e in data['v_vanilla_k10']}

rag = RaptorSubQuestionPipeline()
rag.init()
results = []

for row in TEST_ROWS:
    q = questions[row]
    print(f'\n=== Q{row} ===', flush=True)
    print(f'    {q}', flush=True)
    for use_bilan in [False, True]:
        label = 'AVEC_BILAN' if use_bilan else 'SANS_BILAN'
        print(f'  [{label}] en cours...', flush=True)
        t0 = time.time()
        answer, sources, bilan, sub_qa = rag.query(q, use_bilan=use_bilan)
        elapsed = round(time.time() - t0, 1)
        results.append({
            'row': row,
            'question': q,
            'use_bilan': use_bilan,
            'label': label,
            'answer': answer,
            'bilan': bilan,
            'n_sources': len(sources),
            'elapsed': elapsed,
        })
        print(f'  [{label}] OK ({elapsed}s, {len(sources)} sources)', flush=True)

with open('comparaisons_rag/test_bilan_5q.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\nSauvegardé → comparaisons_rag/test_bilan_5q.json')
