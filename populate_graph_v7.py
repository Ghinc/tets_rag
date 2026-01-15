"""
Script pour peupler le PropertyGraphIndex de v7 depuis les documents

ATTENTION: Ce script va:
1. Lire les documents depuis ChromaDB
2. Extraire automatiquement des triplets (entités + relations) via GPT-3.5
3. Stocker ces triplets dans Neo4j

L'extraction prend du temps (appels LLM pour chaque document).
On limite à 10 documents pour le test initial.

Usage:
    python populate_graph_v7.py
"""

import os
from pathlib import Path
from rag_v7_llamaindex import LlamaIndexRAGPipeline

def load_env():
    """Charge les variables d'environnement depuis .env"""
    env_path = Path(".env")
    if not env_path.exists():
        return {}

    env_vars = {}
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

def main():
    print("="*80)
    print("PEUPLEMENT DU GRAPHE POUR RAG v7")
    print("="*80)

    # Charger les variables depuis .env
    env_vars = load_env()

    # Récupérer la clé OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERREUR: OPENAI_API_KEY non trouvée")
        print("  - Vérifiez que le fichier .env existe")
        print("  - Vérifiez qu'il contient: OPENAI_API_KEY=votre_clé")
        return

    print(f"Clé OpenAI trouvée: {openai_api_key[:10]}...")

    print("\nInitialisation de v7 avec peuplement du graphe...")
    print("ATTENTION: Cette opération va prendre plusieurs minutes")
    print("           (extraction de triplets via GPT-3.5 pour chaque document)")
    print()

    # Initialiser v7 avec populate_graph=True
    # Note: Utilise la base LlamaIndex-RAG sur le port 7688
    rag = LlamaIndexRAGPipeline(
        openai_api_key=openai_api_key,
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        neo4j_uri="bolt://localhost:7688",
        neo4j_user="neo4j",
        neo4j_password="",
        populate_graph=True  # ACTIVER LE PEUPLEMENT
    )

    print("\n" + "="*80)
    print("PEUPLEMENT TERMINÉ")
    print("="*80)
    print("\nLe graphe Neo4j contient maintenant:")
    print("- Les entités extraites des documents")
    print("- Les relations entre ces entités")
    print("\nVous pouvez maintenant relancer le serveur normalement")
    print("avec populate_graph=False (comportement par défaut)")

if __name__ == "__main__":
    main()
