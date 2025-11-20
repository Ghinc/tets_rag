"""
Comparaison avec et sans enrichissement ontologique
"""
import requests
import json

url = "http://localhost:8000/api/query"

question = "Quel est le score Opp à Bastia ?"

print("="*80)
print("COMPARAISON AVEC/SANS ENRICHISSEMENT ONTOLOGIQUE")
print("="*80)
print(f"\nQuestion : {question}\n")

# Test 1 : AVEC enrichissement
print("\n" + "="*80)
print("TEST 1 : AVEC ENRICHISSEMENT ONTOLOGIQUE")
print("="*80)

payload_with = {
    "question": question,
    "rag_version": "v3",
    "k": 5,
    "use_reranking": True,
    "use_ontology_enrichment": True  # ACTIVÉ
}

response_with = requests.post(url, json=payload_with)

if response_with.status_code == 200:
    result_with = response_with.json()
    print(f"\nRéponse : {result_with['answer'][:200]}...")
    print(f"\nSources ({len(result_with['sources'])}) :")
    for i, source in enumerate(result_with['sources'][:3], 1):
        print(f"  {i}. {source['metadata'].get('nom', 'N/A')} - Score: {source['score']:.4f}")
else:
    print(f"Erreur : {response_with.text}")

# Test 2 : SANS enrichissement
print("\n" + "="*80)
print("TEST 2 : SANS ENRICHISSEMENT ONTOLOGIQUE")
print("="*80)

payload_without = {
    "question": question,
    "rag_version": "v3",
    "k": 5,
    "use_reranking": True,
    "use_ontology_enrichment": False  # DÉSACTIVÉ
}

response_without = requests.post(url, json=payload_without)

if response_without.status_code == 200:
    result_without = response_without.json()
    print(f"\nRéponse : {result_without['answer'][:200]}...")
    print(f"\nSources ({len(result_without['sources'])}) :")
    for i, source in enumerate(result_without['sources'][:3], 1):
        print(f"  {i}. {source['metadata'].get('nom', 'N/A')} - Score: {source['score']:.4f}")
else:
    print(f"Erreur : {response_without.text}")

# Test 3 : Utiliser v2 (pas d'ontologie du tout)
print("\n" + "="*80)
print("TEST 3 : V2 (PAS D'ONTOLOGIE)")
print("="*80)

payload_v2 = {
    "question": question,
    "rag_version": "v2",
    "k": 5,
    "use_reranking": True
}

response_v2 = requests.post(url, json=payload_v2)

if response_v2.status_code == 200:
    result_v2 = response_v2.json()
    print(f"\nRéponse : {result_v2['answer'][:200]}...")
    print(f"\nSources ({len(result_v2['sources'])}) :")
    for i, source in enumerate(result_v2['sources'][:3], 1):
        print(f"  {i}. {source['metadata'].get('nom', 'N/A')} - Score: {source['score']:.4f}")
else:
    print(f"Erreur : {response_v2.text}")

print("\n" + "="*80)
print("COMPARAISON COMPLÈTE")
print("="*80)

if response_with.status_code == 200 and response_without.status_code == 200 and response_v2.status_code == 200:
    print("\nCommunes mentionnées :")
    print(f"  AVEC enrichissement : {[s['metadata'].get('nom', 'N/A') for s in result_with['sources'][:5]]}")
    print(f"  SANS enrichissement : {[s['metadata'].get('nom', 'N/A') for s in result_without['sources'][:5]]}")
    print(f"  V2 (pas ontologie) : {[s['metadata'].get('nom', 'N/A') for s in result_v2['sources'][:5]]}")

    print("\nBastia dans les sources ?")
    print(f"  AVEC enrichissement : {'OUI' if any('Bastia' in str(s['metadata'].get('nom', '')) for s in result_with['sources']) else 'NON'}")
    print(f"  SANS enrichissement : {'OUI' if any('Bastia' in str(s['metadata'].get('nom', '')) for s in result_without['sources']) else 'NON'}")
    print(f"  V2 (pas ontologie) : {'OUI' if any('Bastia' in str(s['metadata'].get('nom', '')) for s in result_v2['sources']) else 'NON'}")
