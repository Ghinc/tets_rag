"""
Script de test pour le système RAG multi-version
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_versions():
    """Affiche les versions disponibles"""
    print("\n" + "="*60)
    print("VERSIONS DISPONIBLES")
    print("="*60)
    
    response = requests.get(f"{API_URL}/api/versions")
    versions = response.json()
    
    for v in versions:
        status = "[OK]" if v["available"] else "[NON DISPO]"
        print(f"\n{status} {v['version'].upper()} - {v['name']}")
        print(f"   {v['description']}")
        if v["available"]:
            print("   Features:")
            for feature in v["features"]:
                print(f"     - {feature}")

def test_query(version="v2", question=None):
    """Teste une requête avec une version spécifique"""
    if question is None:
        question = "Quels sont les problèmes de transport à Ajaccio?"
    
    print("\n" + "="*60)
    print(f"TEST REQUÊTE - {version.upper()}")
    print("="*60)
    print(f"Question: {question}")
    
    payload = {
        "question": question,
        "rag_version": version,
        "k": 3,
        "use_reranking": True,
        "include_quantitative": True
    }
    
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[REPONSE] ({result['rag_version_used']})")
            print(result["answer"])
            
            print(f"\n[SOURCES] ({len(result['sources'])} documents)")
            for i, source in enumerate(result['sources'][:3], 1):
                commune = source['metadata'].get('commune', source['metadata'].get('nom', 'N/A'))
                print(f"\n  {i}. {commune} (score: {source['score']:.3f})")
                print(f"     {source['content'][:150]}...")
        else:
            print(f"\n[ERREUR {response.status_code}]")
            print(response.json())
            
    except Exception as e:
        print(f"\n[ERREUR EXCEPTION]")
        print(str(e))

if __name__ == "__main__":
    # Afficher les versions disponibles
    test_versions()
    
    # Tester avec une version disponible
    print("\n" + "="*60)
    input("Appuyez sur Entrée pour tester une requête...")
    
    # Demander quelle version tester
    version = input("\nVersion à tester (v2/v3/v4/v5) [v5]: ").strip() or "v5"
    question = input("Question (ou Entrée pour question par défaut): ").strip()
    
    test_query(version, question if question else None)
