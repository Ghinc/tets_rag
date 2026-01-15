"""
Script de démonstration Graph-RAG

Ce script montre comment utiliser le Graph-RAG avec Neo4j
de manière simple et interactive.

Usage:
    python demo_graphrag.py

Prérequis:
    - Neo4j lancé sur bolt://localhost:7687
    - Fichier .env avec OPENAI_API_KEY (et optionnellement NEO4J_PASSWORD)
"""

import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

from rag_v5_graphrag_neo4j import GraphRAGPipeline


def print_banner():
    """Affiche une bannière de bienvenue"""
    print("="*80)
    print("                    GRAPH-RAG DEMO")
    print("              Retrieval-Augmented Generation avec Neo4j")
    print("="*80)
    print()


def check_environment():
    """Vérifie que l'environnement est correctement configuré"""
    print("[*] Verification de l'environnement...")

    # Vérifier les variables d'environnement
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Neo4j password est optionnelle (peut être None pour connexion sans auth)
    if neo4j_password:
        print("[OK] NEO4J_PASSWORD definie")
    else:
        print("[INFO] NEO4J_PASSWORD non definie (connexion sans authentification)")

    if not openai_key:
        print("[ERREUR] Variable OPENAI_API_KEY non definie")
        print("   Definissez-la avec: export OPENAI_API_KEY=votre_cle")
        sys.exit(1)
    else:
        print("[OK] OPENAI_API_KEY definie")

    print()

    return neo4j_password, openai_key


def initialize_rag(neo4j_password, openai_key):
    """Initialise le pipeline Graph-RAG"""
    print("[*] Initialisation du pipeline Graph-RAG...")
    print()

    try:
        rag = GraphRAGPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            chroma_path="./chroma_v2",
            collection_name="communes_corses_v2",
            openai_api_key=openai_key
        )

        print("[OK] Pipeline initialisé avec succès")
        print()

        return rag

    except Exception as e:
        print(f"[ERREUR] Erreur lors de l'initialisation: {e}")
        print()
        print("Assurez-vous que:")
        print("  1. Neo4j est lancé (http://localhost:7474)")
        print("  2. Le mot de passe Neo4j est correct")
        print("  3. ChromaDB a été créé (lancez rag_v2_improved.py d'abord)")
        sys.exit(1)


def import_data(rag):
    """Importe les données de communes si nécessaire"""
    print("[Stats] Import des données de communes...")

    csv_path = "df_mean_by_commune.csv"

    if not os.path.exists(csv_path):
        print(f"[WARN]  Fichier {csv_path} introuvable. Import des données ignoré.")
        print()
        return

    try:
        rag.import_commune_data(csv_path)
        print("[OK] Données importées dans Neo4j")
        print()

    except Exception as e:
        print(f"[WARN]  Erreur lors de l'import: {e}")
        print("Continuons sans les données quantitatives...")
        print()


def run_demo_queries(rag):
    """Exécute des requêtes de démonstration"""
    print("="*80)
    print("                       REQUÊTES DE DÉMONSTRATION")
    print("="*80)
    print()

    demo_questions = [
        {
            "question": "Quelles sont les principales dimensions du bien-être territorial ?",
            "description": "Test de requête sur l'ontologie"
        },
        {
            "question": "Comment est la santé à Ajaccio ?",
            "description": "Test de fusion graphe + retrieval vectoriel"
        },
        {
            "question": "Quelles communes ont les meilleurs scores en éducation ?",
            "description": "Test de requête quantitative via le graphe"
        }
    ]

    for i, demo in enumerate(demo_questions, 1):
        print(f"[{i}/{len(demo_questions)}] {demo['description']}")
        print(f"[?] Question: {demo['question']}")
        print()

        try:
            response, results = rag.query(
                demo['question'],
                k=3,
                use_graph=True,
                use_reranking=True
            )

            print(f"[=>] Réponse:")
            print(f"{response}")
            print()

            print(f"[Sources] Sources utilisées: {len(results)} documents")
            for j, result in enumerate(results[:3], 1):
                source_type = result.metadata.get('source', 'N/A')
                score = result.score
                print(f"   {j}. {source_type} (score: {score:.3f})")

            print()
            print("-"*80)
            print()

        except Exception as e:
            print(f"[ERREUR] Erreur: {e}")
            print()


def interactive_mode(rag):
    """Mode interactif pour poser des questions"""
    print("="*80)
    print("                        MODE INTERACTIF")
    print("="*80)
    print()
    print("Posez vos questions sur la qualité de vie en Corse.")
    print("Tapez 'quit' ou 'exit' pour quitter.")
    print()

    while True:
        try:
            question = input("[?] Votre question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n[Bye] Au revoir !")
                break

            if not question:
                continue

            print()
            print("[*] Recherche en cours...")
            print()

            response, results = rag.query(
                question,
                k=5,
                use_graph=True,
                use_reranking=True
            )

            print(f"[=>] Réponse:")
            print(f"{response}")
            print()

            print(f"[Sources] {len(results)} sources utilisées")
            print()

        except KeyboardInterrupt:
            print("\n\n[Bye] Au revoir !")
            break

        except Exception as e:
            print(f"[ERREUR] Erreur: {e}")
            print()


def visualize_graph_stats(rag):
    """Affiche des statistiques sur le graphe Neo4j"""
    print("="*80)
    print("                    [Stats] STATISTIQUES DU GRAPHE")
    print("="*80)
    print()

    try:
        with rag.graph.driver.session() as session:
            # Nombre de nœuds par type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS type, count(n) AS count
                ORDER BY count DESC
            """)

            print("Nœuds par type:")
            for record in result:
                print(f"  • {record['type']}: {record['count']}")

            print()

            # Nombre de relations par type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """)

            print("Relations par type:")
            for record in result:
                print(f"  • {record['type']}: {record['count']}")

            print()

    except Exception as e:
        print(f"[WARN]  Impossible de récupérer les statistiques: {e}")
        print()


def main():
    """Fonction principale"""
    print_banner()

    # 1. Vérifier l'environnement
    neo4j_password, openai_key = check_environment()

    # 2. Initialiser le pipeline
    rag = initialize_rag(neo4j_password, openai_key)

    # 3. Importer les données
    import_data(rag)

    # 4. Afficher les statistiques
    visualize_graph_stats(rag)

    # 5. Choisir le mode
    print("Choisissez un mode:")
    print("  1. Démo automatique (3 questions prédéfinies)")
    print("  2. Mode interactif (posez vos propres questions)")
    print()

    choice = input("Votre choix (1 ou 2): ").strip()
    print()

    if choice == "1":
        run_demo_queries(rag)
    elif choice == "2":
        interactive_mode(rag)
    else:
        print("[ERREUR] Choix invalide. Lancement du mode démo par défaut.")
        print()
        run_demo_queries(rag)

    # 6. Fermer les connexions
    print("[*] Fermeture des connexions...")
    rag.close()
    print("[OK] Terminé")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Bye] Interruption par l'utilisateur. Au revoir !")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERREUR] Erreur fatale: {e}")
        sys.exit(1)
