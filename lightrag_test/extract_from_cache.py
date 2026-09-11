"""
Extrait les réponses complètes depuis le cache LightRAG (kv_store_llm_response_cache.json)
sans passer par l'API LightRAG (évite le bug aquery()->None en process frais).
"""
import json, pathlib, re

STORAGE = pathlib.Path(__file__).parent / "lightrag_storage"
CACHE   = STORAGE / "kv_store_llm_response_cache.json"
OUT     = pathlib.Path(__file__).parent / "lightrag_results.json"

cache = json.loads(CACHE.read_text(encoding="utf-8"))

# Les clés du cache sont du type "hybrid:query:<hash>"
query_entries = {k: v for k, v in cache.items() if k.startswith("hybrid:query:")}
print(f"{len(query_entries)} entrées hybrid:query trouvées\n")

# Affiche un aperçu pour identifier les bonnes réponses
for k, v in list(query_entries.items())[:3]:
    content = v.get("return_value", "") or ""
    print(f"  clé: {k}")
    print(f"  début: {content[:120]!r}")
    print()
