"""
Test de la requête OppChoVec sur Bastia
"""
import requests
import json

url = "http://localhost:8000/api/query"

payload = {
    "question": "Quel est le score Opp à Bastia ?",
    "rag_version": "v3",
    "k": 5,
    "use_reranking": True,
    "use_ontology_enrichment": True
}

print("Envoi de la requête...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("\n" + "="*80)
    print("RÉPONSE")
    print("="*80)
    print(result['answer'])

    print("\n" + "="*80)
    print(f"SOURCES ({len(result['sources'])})")
    print("="*80)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. Score: {source['score']:.4f}")
        print(f"   Source: {source['metadata'].get('source', 'unknown')}")
        print(f"   Commune: {source['metadata'].get('nom', source['metadata'].get('commune', 'N/A'))}")
        print(f"   Contenu: {source['content'][:150]}...")

    print("\n" + "="*80)
    print("METADATA")
    print("="*80)
    print(json.dumps(result['metadata'], indent=2))
else:
    print(f"Erreur {response.status_code}: {response.text}")
