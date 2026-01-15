"""
Script pour tester la connexion à Neo4j sur le port 7688
"""

from neo4j import GraphDatabase

uri = "bolt://localhost:7688"
user = "neo4j"
password = ""

print("="*80)
print("TEST CONNEXION NEO4J PORT 7688")
print("="*80)
print(f"URI: {uri}")
print(f"User: {user}")
print(f"Password: {password}")
print()

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        value = result.single()["test"]
        print(f"✅ CONNEXION RÉUSSIE ! Test value: {value}")

        # Compter les nœuds
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"✅ Nombre de nœuds dans la base: {count}")

    driver.close()

except Exception as e:
    print(f"❌ ERREUR DE CONNEXION:")
    print(f"   {e}")
    print()
    print("Vérifiez que:")
    print("  1. La base LlamaIndex-RAG est démarrée dans Neo4j Desktop")
    print("  2. Le port est bien 7688 (Settings → dbms.connector.bolt.listen_address)")
    print("  3. Le mot de passe est bien 'password'")
