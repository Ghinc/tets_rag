import chromadb
import json

client = chromadb.PersistentClient(path='./chroma_v2')
collection = client.get_collection('communes_corses_v2')

results = collection.get(include=['metadatas'])

types = {}
for m in results['metadatas']:
    source = m.get('source', 'unknown')
    types[source] = types.get(source, 0) + 1

print(f"Total documents: {len(results['metadatas'])}")
print("\nRépartition par type de source:")
print(json.dumps(types, indent=2, ensure_ascii=False))

# Regarder un exemple de chaque type
print("\n\nExemple de métadonnées par type:")
seen_types = set()
for m in results['metadatas']:
    source = m.get('source', 'unknown')
    if source not in seen_types:
        seen_types.add(source)
        print(f"\n{source}:")
        print(json.dumps(m, indent=2, ensure_ascii=False))
