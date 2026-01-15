"""
Script pour nettoyer complètement la base Neo4j sur le port 7688
"""

from neo4j import GraphDatabase

uri = "bolt://localhost:7688"
driver = GraphDatabase.driver(uri, auth=("neo4j", ""))

print("="*80)
print("NETTOYAGE COMPLET DE LA BASE NEO4J (PORT 7688)")
print("="*80)
print()

with driver.session() as session:
    # Supprimer toutes les contraintes
    print("1. Suppression des contraintes...")
    result = session.run("SHOW CONSTRAINTS")
    constraints = list(result)
    print(f"   Trouvé {len(constraints)} contraintes")

    for constraint in constraints:
        try:
            constraint_name = constraint.get("name")
            if constraint_name:
                session.run(f"DROP CONSTRAINT {constraint_name}")
                print(f"   - Supprimé: {constraint_name}")
        except Exception as e:
            print(f"   - Erreur: {e}")

    # Supprimer tous les index
    print("\n2. Suppression des index...")
    result = session.run("SHOW INDEXES")
    indexes = list(result)
    print(f"   Trouvé {len(indexes)} index")

    for index in indexes:
        try:
            index_name = index.get("name")
            if index_name:
                session.run(f"DROP INDEX {index_name}")
                print(f"   - Supprimé: {index_name}")
        except Exception as e:
            print(f"   - Erreur: {e}")

    # Supprimer tous les nœuds et relations
    print("\n3. Suppression de tous les nœuds et relations...")
    result = session.run("MATCH (n) DETACH DELETE n")
    print("   OK")

    # Vérifier
    result = session.run("MATCH (n) RETURN count(n) as count")
    count = result.single()["count"]
    print(f"\n4. Vérification: {count} nœuds restants")

driver.close()

print("\n" + "="*80)
print("NETTOYAGE TERMINÉ - Base prête pour le peuplement")
print("="*80)
