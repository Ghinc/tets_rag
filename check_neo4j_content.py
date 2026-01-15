"""
Script pour vérifier le contenu du graphe Neo4j après peuplement
"""

from neo4j import GraphDatabase

# Connexion Neo4j
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", ""))

def check_graph_content():
    with driver.session() as session:
        # Compter les nœuds
        result = session.run("MATCH (n) RETURN count(n) as count")
        node_count = result.single()["count"]
        print(f"Nombre total de nœuds dans Neo4j : {node_count}")

        # Compter les relations
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        print(f"Nombre total de relations dans Neo4j : {rel_count}")

        # Afficher les types de nœuds
        print("\n=== TYPES DE NŒUDS ===")
        result = session.run("""
            MATCH (n)
            RETURN labels(n) as labels, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        for record in result:
            print(f"  {record['labels']}: {record['count']}")

        # Afficher quelques exemples de nœuds
        print("\n=== EXEMPLES DE NŒUDS (10 premiers) ===")
        result = session.run("""
            MATCH (n)
            RETURN labels(n) as labels, properties(n) as props
            LIMIT 10
        """)
        for i, record in enumerate(result, 1):
            print(f"\nNœud {i}:")
            print(f"  Labels: {record['labels']}")
            print(f"  Properties: {record['props']}")

        # Afficher les types de relations
        print("\n=== TYPES DE RELATIONS ===")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        for record in result:
            print(f"  {record['type']}: {record['count']}")

        # Afficher quelques exemples de relations
        print("\n=== EXEMPLES DE RELATIONS (10 premières) ===")
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a) as from_labels, type(r) as rel_type, labels(b) as to_labels
            LIMIT 10
        """)
        for i, record in enumerate(result, 1):
            print(f"\n{i}. {record['from_labels']} --[{record['rel_type']}]--> {record['to_labels']}")

driver.close()

if __name__ == "__main__":
    print("="*80)
    print("CONTENU DU GRAPHE NEO4J")
    print("="*80)
    print()
    check_graph_content()
