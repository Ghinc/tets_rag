"""
Script pour voir les 10 premiers documents de ChromaDB
(ceux qui ont été encodés dans le graphe)
"""

import chromadb

client = chromadb.PersistentClient(path='./chroma_v2')
coll = client.get_collection('communes_corses_v2')

results = coll.get(limit=10, include=['documents', 'metadatas'])

print("="*80)
print("LES 10 PREMIERS DOCUMENTS ENCODÉS DANS LE GRAPHE")
print("="*80)

for i in range(len(results['documents'])):
    metadata = results['metadatas'][i]
    text = results['documents'][i]

    print(f"\n=== DOCUMENT {i+1} ===")
    print(f"Commune: {metadata.get('nom', 'N/A')}")
    print(f"Type: {metadata.get('type', 'N/A')}")
    print(f"Source: {metadata.get('source', 'N/A')}")
    print(f"Texte (200 premiers caractères):")
    print(text[:200])
    print("...")
