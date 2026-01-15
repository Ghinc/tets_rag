# 🔗 Graph-RAG : Guide complet

## 🎯 Réponse rapide : Oui, je sais connecter votre code à Neo4j !

J'ai créé **3 implémentations complètes** de Graph-RAG pour votre projet :

1. **RAG v5** ([rag_v5_graphrag_neo4j.py](rag_v5_graphrag_neo4j.py)) - Graph-RAG avec LangChain + Neo4j
2. **RAG v6** ([rag_v6_gretriever.py](rag_v6_gretriever.py)) - G-Retriever avec GNN (PyTorch Geometric)
3. **Demo** ([demo_graphrag.py](demo_graphrag.py)) - Script interactif pour tester

---

## 🚀 Démarrage rapide (5 minutes)

### 1. Installer Neo4j

**Docker (plus simple) :**
```bash
docker run --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    -d neo4j:latest
```

**Vérifier :** Ouvrir http://localhost:7474 (connexion: neo4j/password123)

### 2. Installer les dépendances

```bash
pip install -r requirements_graphrag.txt
```

### 3. Configurer l'environnement

```bash
export NEO4J_PASSWORD=password123
export OPENAI_API_KEY=votre_clé
```

### 4. Lancer la démo

```bash
python demo_graphrag.py
```

**C'est tout !** 🎉

---

## 🔄 Vous avez déjà Neo4j avec votre ontologie ?

**Encore plus simple !** Le code détecte automatiquement si l'ontologie existe :

```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="votre_password",
    openai_api_key="votre_clé"
)
# ✓ L'ontologie existante est automatiquement détectée et réutilisée !
# Pas besoin de réimporter

response, results = rag.query("Votre question ?")
```

**Voir le guide détaillé :** [QUICKSTART_NEO4J_EXISTANT.md](QUICKSTART_NEO4J_EXISTANT.md)

---

## 📊 Qu'est-ce que le Graph-RAG ?

### RAG Classique (v1-v4)
```
Question → Embedding → Vector Search → Chunks → LLM → Réponse
                ↓
           ChromaDB (vecteurs)
```

### Graph-RAG (v5-v6)
```
Question → Embedding → Hybrid Search → Context → LLM → Réponse
                ↓              ↓
           ChromaDB    +    Neo4j
           (vecteurs)      (graphe)
                              ↓
                    Relations explicites
                    Raisonnement multi-sauts
```

---

## 🔍 Avantages du Graph-RAG pour votre projet

### 1. **Exploitation de votre ontologie**

**Sans Graph-RAG (v3) :**
```
Question: "Quelles dimensions influencent la santé ?"
→ Enrichissement textuel : "santé medical hôpital Vec3..."
→ Recherche vectorielle
```

**Avec Graph-RAG (v5) :**
```
Question: "Quelles dimensions influencent la santé ?"
→ Requête Cypher :
   MATCH (d:Dimension)-[:INFLUENCES]->(h:Health)
   RETURN d.label
→ Relations explicites du graphe
```

### 2. **Raisonnement multi-sauts**

**Exemple :** "Pourquoi Ajaccio a un score Vec bas ?"

**Graph-RAG peut raisonner :**
```
Ajaccio → HAS_INDICATOR_VALUE → Vec3 (bas)
       ↓
    CAUSED_BY
       ↓
Health dimension → MEASURED_BY → Nombre de médecins (bas)
       ↓
    CORRELATES_WITH
       ↓
Services dimension → Transport insuffisant
```

### 3. **Visualisation interactive**

Ouvrez Neo4j Browser (http://localhost:7474) et explorez visuellement :
```cypher
// Voir tout le graphe
MATCH (n) RETURN n LIMIT 100

// Voir les relations d'une commune
MATCH path = (c:Commune {name: "Ajaccio"})-[*1..3]-(x)
RETURN path
```

---

## 🆚 Comparaison avec G-Retriever

| Aspect | RAG v5 (Neo4j + Cypher) | RAG v6 (G-Retriever + GNN) |
|--------|-------------------------|----------------------------|
| **Setup** | Simple | Complexe (PyTorch, GPU) |
| **Requêtes** | Cypher (SQL-like) | GNN embeddings |
| **Explicabilité** | ✅ Excellente (chemins visibles) | ⚠️ Boîte noire (NN) |
| **Personnalisation** | ✅ Facile (queries Cypher) | ❌ Difficile (fine-tuning) |
| **Performance (petits graphes)** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance (gros graphes >100k)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production** | ✅ Prêt | ❌ Expérimental |

**Recommandation :** Commencer avec v5, passer à v6 seulement si nécessaire.

---

## 💡 Cas d'usage concrets pour votre thèse

### 1. Explorer les corrélations

```python
# Trouver les indicateurs corrélés à Ajaccio
rag.graph.driver.session().run("""
    MATCH (c:Commune {name: "Ajaccio"})-[:HAS_INDICATOR_VALUE]->(iv1)
    MATCH (c)-[:HAS_INDICATOR_VALUE]->(iv2)
    WHERE iv1.indicator_name CONTAINS "opp"
      AND iv2.indicator_name CONTAINS "vec"
    RETURN iv1.value, iv2.value
""")
```

### 2. Identifier les dimensions influentes

```python
# Quelles dimensions sont liées à l'éducation ?
dimensions = rag.graph.find_related_dimensions(["education", "école"])
```

### 3. Comparer des communes

```python
# Communes similaires à Ajaccio (par indicateurs)
rag.graph.driver.session().run("""
    MATCH (c1:Commune {name: "Ajaccio"})-[:HAS_INDICATOR_VALUE]->(iv1)
    MATCH (c2:Commune)-[:HAS_INDICATOR_VALUE]->(iv2)
    WHERE c1 <> c2
      AND iv1.indicator_name = iv2.indicator_name
      AND abs(iv1.value - iv2.value) < 5
    RETURN c2.name, count(*) AS similar_indicators
    ORDER BY similar_indicators DESC
    LIMIT 10
""")
```

---

## 📈 Architecture de votre système Graph-RAG

```
┌─────────────────────────────────────────────────────────────┐
│                         UTILISATEUR                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAGPipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Query Parser │→ │  Enricher    │→ │   Fusion     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
             ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    ChromaDB     │ │     Neo4j       │ │   OpenAI API    │
│  (Vecteurs)     │ │   (Graphe)      │ │     (LLM)       │
│                 │ │                 │ │                 │
│ - Entretiens    │ │ - Ontologie     │ │ - GPT-3.5/4     │
│ - Verbatims     │ │ - Communes      │ │ - Generation    │
│ - Wikipedia     │ │ - Indicateurs   │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 🔧 Personnalisation avancée

### Ajouter des relations géographiques

```python
# Dans Neo4j : créer des relations de proximité
session.run("""
    MATCH (c1:Commune), (c2:Commune)
    WHERE c1.name <> c2.name
      AND point.distance(
            point({latitude: c1.lat, longitude: c1.lon}),
            point({latitude: c2.lat, longitude: c2.lon})
          ) < 50000
    MERGE (c1)-[:NEAR {distance_km:
        point.distance(...)/1000
    }]->(c2)
""")
```

### Ajouter des entretiens comme nœuds

```python
# Chaque entretien devient un nœud
for interview_id, interview_data in entretiens.items():
    session.run("""
        CREATE (e:Interview {
            id: $id,
            text: $text,
            date: $date
        })
        MATCH (c:Commune {name: $commune})
        MERGE (e)-[:FROM_COMMUNE]->(c)
    """,
    id=interview_id,
    text=interview_data['text'],
    commune=interview_data['commune'])
```

### Requêtes analytiques personnalisées

```python
class CustomGraphQueries:
    """Requêtes personnalisées pour votre thèse"""

    def find_low_vec_causes(self, commune: str) -> List[str]:
        """Trouve les causes d'un score Vec bas"""
        result = session.run("""
            MATCH (c:Commune {name: $commune})-[:HAS_INDICATOR_VALUE]->(vec)
            WHERE vec.indicator_name CONTAINS "Vec"
              AND vec.value < 50
            MATCH (c)-[:HAS_INDICATOR_VALUE]->(related)
            WHERE related.value < 50
            RETURN related.indicator_name AS cause, related.value
            ORDER BY related.value ASC
        """, commune=commune)

        return [dict(r) for r in result]
```

---

## 📚 Fichiers créés pour vous

| Fichier | Description |
|---------|-------------|
| [rag_v5_graphrag_neo4j.py](rag_v5_graphrag_neo4j.py) | **Implémentation Graph-RAG principale** |
| [rag_v6_gretriever.py](rag_v6_gretriever.py) | Implémentation G-Retriever (GNN) |
| [demo_graphrag.py](demo_graphrag.py) | **Script de démonstration interactif** |
| [GRAPH_RAG_SETUP.md](GRAPH_RAG_SETUP.md) | Guide d'installation détaillé |
| [COMPARAISON_RAG_VERSIONS.md](COMPARAISON_RAG_VERSIONS.md) | **Comparaison toutes versions** |
| [requirements_graphrag.txt](requirements_graphrag.txt) | Dépendances Python |

---

## 🎓 Pour votre thèse

### Contributions possibles

1. **Comparaison empirique**
   - Benchmark RAG classique vs Graph-RAG
   - Sur votre dataset de communes corses
   - Métriques : précision, temps, coût

2. **Extension de l'ontologie**
   - Enrichir l'ontologie avec relations causales
   - Apprendre les relations depuis les données
   - Validation par experts

3. **Interface de visualisation**
   - Dashboard Neo4j Browser customisé
   - Visualisation des chemins de raisonnement
   - Export des graphes pour publications

### Publications potentielles

- "Graph-RAG pour l'analyse territoriale : application au bien-être en Corse"
- "Comparaison des approches RAG pour l'analyse de données qualitatives et quantitatives"
- "Ontologie du bien-être territorial : de la modélisation au Graph-RAG"

---

## 🐛 Troubleshooting

### Neo4j ne se connecte pas
```bash
# Vérifier que Neo4j est lancé
docker ps | grep neo4j

# Vérifier les logs
docker logs neo4j
```

### Erreur "Graph is empty"
```python
# Réimporter l'ontologie
rag.graph.clear_database()  # ATTENTION : efface tout !
rag.graph.import_ontology(rag.ontology_parser)
```

### Performance lente
```cypher
// Créer des index
CREATE INDEX commune_name IF NOT EXISTS FOR (c:Commune) ON (c.name)
CREATE INDEX indicator_name IF NOT EXISTS FOR (iv:IndicatorValue) ON (iv.indicator_name)
```

---

## 📞 Support

- **Documentation Neo4j :** https://neo4j.com/docs/
- **LangChain Graph :** https://python.langchain.com/docs/integrations/graphs/
- **G-Retriever Paper :** https://arxiv.org/abs/2402.07630

**Besoin d'aide ?** Consultez les guides ci-dessus ou ouvrez une issue !

---

## ✅ Prochaines étapes recommandées

1. **[ ] Installer Neo4j** (5 min avec Docker)
2. **[ ] Lancer demo_graphrag.py** (tester avec vos données)
3. **[ ] Explorer le graphe** (Neo4j Browser)
4. **[ ] Comparer v2 vs v5** (benchmark sur vos questions)
5. **[ ] Adapter à votre use case** (requêtes Cypher personnalisées)

**Bonne chance avec votre Graph-RAG ! 🚀**
