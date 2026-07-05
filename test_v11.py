import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from rag_v11_agentic import AgenticRAGPipeline

p = AgenticRAGPipeline(chroma_path='./chroma_portrait')
p.init()
print('Init OK', flush=True)

answer, sources, sub_qs = p.query(
    'Quel est le score moyen de bien-etre a Ajaccio ?',
    use_fast_path=False,
)
print()
print('=== REPONSE ===')
print(answer)
print(f'({len(sources)} sources, {len(sub_qs) if sub_qs else 0} sous-questions)')
