"""
Script pour lister TOUS les labels de nœuds dans Neo4j
"""

from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", ""))

with driver.session() as session:
    # Tous les labels
    result = session.run("CALL db.labels()")
    labels = [record[0] for record in result]

    print("="*80)
    print(f"TOUS LES LABELS DANS NEO4J ({len(labels)} labels)")
    print("="*80)
    for label in sorted(labels):
        print(f"  - {label}")

    print("\n" + "="*80)
    print("NOMBRE DE NŒUDS PAR LABEL")
    print("="*80)
    for label in sorted(labels):
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"  {label}: {count}")

driver.close()
