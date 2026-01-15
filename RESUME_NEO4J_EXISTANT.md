# ✨ Résumé : Utiliser le Graph-RAG avec votre Neo4j existant

## 🎯 TL;DR (Trop Long ; Pas Lu)

Si vous avez **déjà Neo4j + votre ontologie importée** :

```bash
# 1. Installer juste les dépendances Python
pip install neo4j chromadb openai sentence-transformers

# 2. Tester la connexion
export NEO4J_PASSWORD=votre_password
python test_neo4j_connection.py

# 3. Utiliser directement
python demo_graphrag.py
```

**Le code détecte automatiquement votre ontologie existante !** ✅

---

## 🔧 Ce qui change dans la marche à suivre

### Avant (sans Neo4j existant)

1. ❌ Installer Neo4j
2. ❌ Importer l'ontologie .ttl dans Neo4j
3. ✅ Installer les dépendances Python
4. ✅ Lancer le code

### Maintenant (avec Neo4j existant)

1. ✅ ~~Installer Neo4j~~ → **Déjà fait !**
2. ✅ ~~Importer l'ontologie~~ → **Déjà fait !**
3. ✅ Installer les dépendances Python
4. ✅ Lancer le code → **Il détecte automatiquement l'ontologie**

**Vous sautez 2 étapes ! 🚀**

---

## 🔍 Détection automatique

Le code a été modifié pour **détecter automatiquement** si l'ontologie existe :

```python
# Dans rag_v5_graphrag_neo4j.py (ligne 77-92)
def check_ontology_exists(self) -> bool:
    """Vérifie si l'ontologie est déjà dans Neo4j"""
    with self.driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension', 'Indicator'])
            RETURN count(n) AS count
        """)
        return count > 0

def import_ontology(self, ontology_parser, force_reimport=False):
    """Import avec détection automatique"""
    if not force_reimport and self.check_ontology_exists():
        print("✓ Ontologie déjà présente dans Neo4j (import skippé)")
        return  # Ne fait rien !

    # Sinon, importe normalement...
```

---

## 📋 Checklist rapide

### Avant de lancer le code

- [ ] Neo4j est lancé (`http://localhost:7474` accessible)
- [ ] Vous avez vos credentials (user/password)
- [ ] L'ontologie est visible dans Neo4j Browser
- [ ] Variables d'environnement définies :
  ```bash
  export NEO4J_PASSWORD=votre_password
  export OPENAI_API_KEY=votre_clé
  ```

### Test rapide

```bash
# 1. Tester la connexion et vérifier l'ontologie
python test_neo4j_connection.py
```

**Ce script vous dit :**
- ✅ Si la connexion fonctionne
- ✅ Si l'ontologie est détectée
- ✅ Nombre de nœuds et relations
- ✅ Performances de base

### Lancer le Graph-RAG

```bash
# 2. Lancer la démo
python demo_graphrag.py
```

**Ou dans votre code :**

```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="votre_password",
    openai_api_key="votre_clé"
)

# L'ontologie est automatiquement détectée et réutilisée !

response, results = rag.query(
    "Quelles dimensions du bien-être sont liées à la santé ?",
    use_graph=True
)
print(response)
```

---

## 🎯 Cas particuliers

### Si vos labels sont différents

Par exemple, si vous avez `Theme` au lieu de `Dimension` :

```python
# Modifier la requête de détection (ligne 85-89)
result = session.run("""
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Theme', 'Indicator'])
    --                                              ↑ Changez ici
    RETURN count(n) AS count
""")
```

### Si vous voulez forcer le réimport

```python
# Forcer le réimport même si l'ontologie existe
rag.graph.import_ontology(rag.ontology_parser, force_reimport=True)
```

### Si vous avez plusieurs databases

```python
# Spécifier la database
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    database="ma_database"  # ← Ajouter ce paramètre
)
```

---

## 💡 Avantages de cette approche

✅ **Pas de duplication** - Une seule ontologie, dans Neo4j
✅ **Modifications persistantes** - Vos changements dans Neo4j sont directement utilisés
✅ **Pas de resynchronisation** - Toujours à jour
✅ **Performance** - Pas de temps d'import à chaque démarrage (gain de ~15-30 secondes)

---

## 📊 Workflow recommandé

```
┌─────────────────────────────────────────────────────────────┐
│                  VOTRE NEO4J EXISTANT                        │
│  ┌──────────────────────────────────────────────┐           │
│  │ Ontologie du bien-être                       │           │
│  │  - Concepts (WellBeing, QualityOfLife, ...) │           │
│  │  - Dimensions (Health, Housing, ...)        │           │
│  │  - Indicateurs (Opp1, Vec3, ...)            │           │
│  └──────────────────────────────────────────────┘           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              GRAPH-RAG PIPELINE (v5)                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ 1. Connexion à Neo4j                         │           │
│  │ 2. Détection automatique de l'ontologie ✓   │           │
│  │ 3. Import données communes (si nécessaire)   │           │
│  │ 4. Prêt à répondre aux questions !           │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Commandes essentielles

### Tester la connexion
```bash
python test_neo4j_connection.py
```

### Lancer la démo
```bash
python demo_graphrag.py
```

### Utiliser dans votre code
```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    openai_api_key="key"
)

response, _ = rag.query("Ma question ?")
```

### Vérifier l'ontologie dans Neo4j Browser
```cypher
// http://localhost:7474

// Voir l'ontologie
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension', 'Indicator'])
RETURN n
LIMIT 100

// Voir les relations
MATCH (n)-[r]->(m)
WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension'])
RETURN n, r, m
LIMIT 50
```

---

## 🎓 Pour aller plus loin

### Ajouter vos propres requêtes

```python
# Requête personnalisée sur votre graphe
with rag.graph.driver.session() as session:
    result = session.run("""
        MATCH (c:Commune {name: $commune})-[*1..3]-(x)
        RETURN DISTINCT x.label AS related_concepts
    """, commune="Ajaccio")

    for record in result:
        print(record['related_concepts'])
```

### Enrichir le graphe

```python
# Ajouter des relations entre communes
session.run("""
    MATCH (c1:Commune), (c2:Commune)
    WHERE c1.name <> c2.name
      AND distance(...) < 50000
    MERGE (c1)-[:NEAR]->(c2)
""")
```

---

## 📚 Fichiers de référence

| Fichier | Utilité |
|---------|---------|
| [test_neo4j_connection.py](test_neo4j_connection.py) | **Tester la connexion** |
| [demo_graphrag.py](demo_graphrag.py) | **Démo interactive** |
| [rag_v5_graphrag_neo4j.py](rag_v5_graphrag_neo4j.py) | Code principal |
| [QUICKSTART_NEO4J_EXISTANT.md](QUICKSTART_NEO4J_EXISTANT.md) | Guide détaillé |
| [README_GRAPHRAG.md](README_GRAPHRAG.md) | Vue d'ensemble |

---

## ❓ FAQ

**Q: Le code va-t-il modifier mon ontologie existante ?**
R: Non ! Par défaut, il détecte l'ontologie existante et la réutilise sans modification.

**Q: Puis-je quand même forcer le réimport ?**
R: Oui, avec `force_reimport=True`, mais attention, cela écrasera l'existant.

**Q: Mes labels sont différents (Theme au lieu de Dimension), ça marche ?**
R: Il faut adapter la requête de détection (voir "Cas particuliers" ci-dessus).

**Q: Le code fonctionne-t-il avec Neo4j Desktop ?**
R: Oui ! Utilisez l'URI affichée dans Neo4j Desktop (généralement `bolt://localhost:7687`).

**Q: Puis-je utiliser plusieurs databases ?**
R: Oui, spécifiez le paramètre `database` lors de la connexion.

---

## ✅ Résumé final

**Avec Neo4j existant :**
1. ✅ Installer dépendances Python → `pip install neo4j chromadb openai`
2. ✅ Tester connexion → `python test_neo4j_connection.py`
3. ✅ Lancer démo → `python demo_graphrag.py`
4. ✅ L'ontologie est automatiquement détectée et réutilisée !

**Gain de temps : ~20-30 minutes** (pas besoin d'installer Neo4j ni d'importer l'ontologie)

**Prêt à utiliser votre Graph-RAG ! 🎉**
