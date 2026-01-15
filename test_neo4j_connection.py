"""
Script de test pour vérifier la connexion à Neo4j et l'état de l'ontologie

Usage:
    python test_neo4j_connection.py

Ce script vérifie :
1. La connexion à Neo4j
2. L'existence de l'ontologie
3. La structure du graphe
4. Les performances de base

Auteur: Claude Code
Date: 2025-01-04
"""

import os
import sys
from typing import Optional
from neo4j import GraphDatabase
from tabulate import tabulate


def print_banner():
    """Affiche une bannière"""
    print("="*80)
    print("               [*] TEST DE CONNEXION NEO4J + ONTOLOGIE")
    print("="*80)
    print()


def test_connection(uri: str, user: str, password: Optional[str]) -> bool:
    """
    Test de connexion à Neo4j

    Returns:
        True si connexion OK, False sinon
    """
    print("[*] Test de connexion à Neo4j...")
    print(f"   URI: {uri}")
    print(f"   User: {user}")

    if password is None or password == "":
        print("   Auth: Sans authentification")
    else:
        print("   Auth: Avec authentification")
    print()

    try:
        if password is None or password == "":
            driver = GraphDatabase.driver(uri)
        else:
            driver = GraphDatabase.driver(uri, auth=(user, password))

        driver.verify_connectivity()
        print("[OK] Connexion réussie !")
        print()
        return True, driver

    except Exception as e:
        print(f"[ERREUR] Échec de connexion: {e}")
        print()
        print("Vérifiez que:")
        print("  1. Neo4j est lancé")
        print("  2. L'URI est correcte (bolt://localhost:7687)")
        print("  3. Si auth activée, le mot de passe est correct")
        print()
        return False, None


def check_ontology(driver) -> dict:
    """
    Vérifie l'état de l'ontologie dans Neo4j

    Returns:
        Dict avec les statistiques
    """
    print("[*] Analyse de l'ontologie...")
    print()

    stats = {}

    with driver.session() as session:
        # 1. Compter les nœuds par type
        result = session.run("""
            MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count
            ORDER BY count DESC
        """)

        nodes_by_type = []
        total_nodes = 0
        for record in result:
            labels = record['labels']
            count = record['count']
            total_nodes += count

            label_str = ":".join(labels) if labels else "No label"
            nodes_by_type.append([label_str, count])

        stats['total_nodes'] = total_nodes
        stats['nodes_by_type'] = nodes_by_type

        # 2. Compter les relations par type
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)

        relations_by_type = []
        total_relations = 0
        for record in result:
            rel_type = record['type']
            count = record['count']
            total_relations += count
            relations_by_type.append([rel_type, count])

        stats['total_relations'] = total_relations
        stats['relations_by_type'] = relations_by_type

        # 3. Vérifier spécifiquement l'ontologie
        # Détection flexible : labels OWL (neosemantics) OU labels personnalisés
        result = session.run("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE
                label IN ['Concept', 'Dimension', 'Indicator'] OR
                label STARTS WITH 'owl__' OR
                label = 'Resource'
            )
            RETURN count(n) AS ontology_nodes
        """)

        record = result.single()
        stats['ontology_nodes'] = record['ontology_nodes'] if record else 0

    return stats


def display_stats(stats: dict):
    """Affiche les statistiques du graphe"""

    print("[Stats] STATISTIQUES DU GRAPHE")
    print("-" * 80)
    print()

    # Résumé
    print(f"Total de nœuds: {stats['total_nodes']}")
    print(f"Total de relations: {stats['total_relations']}")
    print(f"Nœuds d'ontologie (Concept/Dimension/Indicator): {stats['ontology_nodes']}")
    print()

    # Nœuds par type
    if stats['nodes_by_type']:
        print("Nœuds par type:")
        print(tabulate(stats['nodes_by_type'], headers=["Type", "Count"], tablefmt="grid"))
        print()

    # Relations par type
    if stats['relations_by_type']:
        print("Relations par type:")
        print(tabulate(stats['relations_by_type'], headers=["Type", "Count"], tablefmt="grid"))
        print()


def check_ontology_status(stats: dict):
    """Vérifie l'état de l'ontologie et donne des recommandations"""

    print("[*] ÉTAT DE L'ONTOLOGIE")
    print("-" * 80)
    print()

    if stats['ontology_nodes'] > 0:
        print(f"[OK] Ontologie détectée ({stats['ontology_nodes']} nœuds)")
        print()
        print("Votre ontologie est déjà importée dans Neo4j.")
        print("Le code Graph-RAG va automatiquement la détecter et la réutiliser.")
        print()
        print("👉 Vous pouvez lancer directement:")
        print("   python demo_graphrag.py")
        print()

    else:
        print("[WARN]  Aucune ontologie détectée")
        print()
        print("Aucun nœud avec les labels Concept, Dimension ou Indicator trouvé.")
        print()
        print("Options:")
        print("  1. Vérifier que l'ontologie est bien importée")
        print("  2. Vérifier que les labels correspondent (Concept, Dimension, Indicator)")
        print("  3. Laisser le code Graph-RAG l'importer automatiquement")
        print()


def test_query_performance(driver):
    """Test de performance d'une requête simple"""

    print("[*] TEST DE PERFORMANCE")
    print("-" * 80)
    print()

    import time

    with driver.session() as session:
        # Test 1: Requête simple
        start = time.time()
        result = session.run("MATCH (n) RETURN count(n) AS count")
        record = result.single()
        elapsed = time.time() - start

        print(f"Requête: MATCH (n) RETURN count(n)")
        print(f"Résultat: {record['count']} nœuds")
        print(f"Temps: {elapsed*1000:.2f} ms")
        print()

        # Test 2: Requête avec filtre
        if record['count'] > 0:
            start = time.time()
            result = session.run("""
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension'])
                RETURN count(n) AS count
            """)
            record = result.single()
            elapsed = time.time() - start

            print(f"Requête: Filtrer par Concept/Dimension")
            print(f"Résultat: {record['count']} nœuds")
            print(f"Temps: {elapsed*1000:.2f} ms")
            print()


def suggest_next_steps(stats: dict):
    """Suggère les prochaines étapes"""

    print("[*] PROCHAINES ÉTAPES")
    print("-" * 80)
    print()

    if stats['ontology_nodes'] > 0:
        print("1. [ ] Lancer demo_graphrag.py pour tester le Graph-RAG")
        print("2. [ ] Importer les données de communes (si pas encore fait)")
        print("3. [ ] Poser des questions sur votre ontologie")
        print()
        print("Commande rapide:")
        print("  export NEO4J_PASSWORD=votre_password")
        print("  export OPENAI_API_KEY=votre_clé")
        print("  python demo_graphrag.py")
        print()

    else:
        print("1. [ ] Vérifier que l'ontologie .ttl est bien importée dans Neo4j")
        print("2. [ ] Ou laisser le Graph-RAG l'importer automatiquement")
        print("3. [ ] Lancer demo_graphrag.py")
        print()


def main():
    """Fonction principale"""

    print_banner()

    # 1. Récupérer les credentials
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if not neo4j_password:
        print("[INFO]  Variable NEO4J_PASSWORD non définie")
        print()
        response = input("Neo4j avec authentification ? (o/n): ").strip().lower()
        if response == 'o':
            neo4j_password = input("Entrez le mot de passe Neo4j: ").strip()
        else:
            neo4j_password = None
            print("Mode sans authentification")
        print()

    # 2. Test de connexion
    success, driver = test_connection(neo4j_uri, neo4j_user, neo4j_password)

    if not success:
        sys.exit(1)

    try:
        # 3. Analyser l'ontologie
        stats = check_ontology(driver)

        # 4. Afficher les statistiques
        display_stats(stats)

        # 5. Vérifier l'état de l'ontologie
        check_ontology_status(stats)

        # 6. Test de performance
        test_query_performance(driver)

        # 7. Suggestions
        suggest_next_steps(stats)

    except Exception as e:
        print(f"[ERREUR] Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.close()
        print("[*] Connexion fermée")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Bye] Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERREUR] Erreur fatale: {e}")
        sys.exit(1)
