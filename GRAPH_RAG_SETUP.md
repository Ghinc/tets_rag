# 🚀 Guide de mise en place du Graph-RAG avec Neo4j

## 📋 Prérequis

### 1. Installer Neo4j

**Option A : Neo4j Desktop (Recommandé pour débuter)**
```bash
# Télécharger depuis https://neo4j.com/download/
# Lancer Neo4j Desktop
# Créer une nouvelle base de données
# Définir un mot de passe (noter le mot de passe !)
```

**Option B : Neo4j Docker (Recommandé pour production)**
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/votre_mot_de_passe \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

**Option C : Neo4j Community Edition (Linux/Mac)**
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Démarrer Neo4j
sudo systemctl start neo4j
```

### 2. Installer les dépendances Python

```bash
pip install neo4j>=5.0.0
pip install langchain-neo4j
pip install graph-rag  # Pour G-Retriever (optionnel)
```

### 3. Configuration des variables d'environnement

```bash
# .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
OPENAI_API_KEY=votre_clé_openai
```

---

## 🎯 Utilisation du Graph-RAG

### Démarrage basique

```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

# Initialiser
rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="votre_mot_de_passe",
    openai_api_key="votre_clé"
)

# Importer les données de communes
rag.import_commune_data("df_mean_by_commune.csv")

# Poser une question
response, results = rag.query(
    "Comment est la santé à Ajaccio ?",
    use_graph=True  # Active l'enrichissement par graphe
)

print(response)
```

### Visualiser le graphe dans Neo4j Browser

1. Ouvrir http://localhost:7474
2. Se connecter avec `neo4j` / `votre_mot_de_passe`
3. Exécuter des requêtes Cypher :

```cypher
// Voir tous les nœuds et relations
MATCH (n) RETURN n LIMIT 100

// Voir les dimensions
MATCH (d:Dimension) RETURN d

// Voir une commune et ses indicateurs
MATCH (c:Commune {name: "Ajaccio"})-[:HAS_INDICATOR_VALUE]->(iv)
RETURN c, iv

// Trouver les communes avec les meilleurs scores en santé
MATCH (c:Commune)-[:HAS_INDICATOR_VALUE]->(iv:IndicatorValue)
WHERE iv.indicator_name CONTAINS "sante"
RETURN c.name, iv.value
ORDER BY iv.value DESC
LIMIT 10
```

---

## 🔬 Avantages du Graph-RAG vs RAG classique

| Aspect | RAG Classique (v1-v4) | Graph-RAG (v5) |
|--------|----------------------|----------------|
| **Structure** | Documents plats | Graphe de connaissances |
| **Relations** | Implicites (embeddings) | Explicites (arêtes) |
| **Ontologie** | Enrichissement textuel | Structure native |
| **Raisonnement** | Similarité vectorielle | Traversée de graphe + vecteurs |
| **Explicabilité** | Scores de similarité | Chemins dans le graphe |

### Cas d'usage où Graph-RAG excelle :

1. **Questions à multiples sauts**
   - ❌ RAG classique : "Quelle dimension influence l'emploi ?"
   - ✅ Graph-RAG : Traverse `Emploi -> Dimension -> Indicateurs -> Communes`

2. **Raisonnement relationnel**
   - ❌ RAG classique : Difficulté à lier causalement
   - ✅ Graph-RAG : Relations explicites (CAUSES, INFLUENCED_BY, etc.)

3. **Données structurées + non structurées**
   - ❌ RAG classique : Deux systèmes séparés
   - ✅ Graph-RAG : Fusion native dans le graphe

---

## 🧪 Comparaison avec G-Retriever

**G-Retriever** est un modèle de Graph Neural Network (GNN) pour le retrieval.

### Architecture G-Retriever

```
Question → Embedding → GNN Encoder → Subgraph Retrieval → LLM
                ↓                            ↓
              Neo4j Graph ----------------→  Context
```

### Différences clés

| Méthode | Rag v5 (LangChain + Neo4j) | G-Retriever |
|---------|----------------------------|-------------|
| Approche | Requêtes Cypher + Retrieval vectoriel | GNN pour encoder le graphe |
| Complexité | Simple, modulaire | Plus complexe, end-to-end |
| Flexibilité | Facile à personnaliser | Nécessite fine-tuning |
| Performance | Bonne pour graphes moyens | Excellente pour gros graphes |

---

## 📊 Schéma du graphe créé

```
(Concept:WellBeing)
    ├─[:HAS_DIMENSION]→ (Dimension:Health)
    │                       ├─[:MEASURED_BY]→ (Indicator:Vec3)
    │                       └─[:APPLIES_TO]→ (Commune:Ajaccio)
    │                                           └─[:HAS_INDICATOR_VALUE]→ (IndicatorValue {value: 75.3})
    │
    ├─[:HAS_DIMENSION]→ (Dimension:Education)
    └─[:HAS_DIMENSION]→ (Dimension:Housing)
```

---

## 🚀 Prochaines étapes

### 1. Enrichir le graphe avec plus de relations

```python
# Ajouter des relations entre communes (proximité géographique)
session.run("""
    MATCH (c1:Commune), (c2:Commune)
    WHERE c1.name <> c2.name
      AND distance(point({latitude: c1.lat, longitude: c1.lon}),
                   point({latitude: c2.lat, longitude: c2.lon})) < 50000
    MERGE (c1)-[:NEAR {distance_km: distance(...)/1000}]->(c2)
""")
```

### 2. Ajouter des entretiens comme nœuds

```python
# Chaque entretien devient un nœud relié à la commune
session.run("""
    CREATE (e:Interview {id: $id, text: $text, date: $date})
    MATCH (c:Commune {name: $commune})
    MERGE (e)-[:FROM_COMMUNE]->(c)
""", id=interview_id, text=interview_text, commune=commune_name)
```

### 3. Implémenter des requêtes de raisonnement

```python
def find_correlated_indicators(self, commune: str) -> List[str]:
    """Trouve les indicateurs corrélés pour une commune"""
    result = session.run("""
        MATCH (c:Commune {name: $commune})-[:HAS_INDICATOR_VALUE]->(iv1)
        MATCH (c)-[:HAS_INDICATOR_VALUE]->(iv2)
        WHERE iv1 <> iv2
          AND abs(iv1.value - iv2.value) < 10
        RETURN iv1.indicator_name AS ind1,
               iv2.indicator_name AS ind2,
               abs(iv1.value - iv2.value) AS diff
        ORDER BY diff ASC
    """, commune=commune)
    return list(result)
```

---

## 📚 Ressources

- [Neo4j Graph Database](https://neo4j.com/)
- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher)
- [G-Retriever Paper](https://arxiv.org/abs/2402.07630)
- [GraphRAG by Microsoft](https://github.com/microsoft/graphrag)

---

## 🐛 Troubleshooting

### "Connection refused" lors de la connexion Neo4j
```bash
# Vérifier que Neo4j est lancé
sudo systemctl status neo4j

# Vérifier les ports
netstat -an | grep 7687
```

### "Authentication failed"
```bash
# Réinitialiser le mot de passe Neo4j
neo4j-admin set-initial-password nouveau_mot_de_passe
```

### Performance lente sur gros graphes
```cypher
// Créer des index sur les propriétés fréquemment recherchées
CREATE INDEX commune_name IF NOT EXISTS FOR (c:Commune) ON (c.name)
CREATE INDEX indicator_name IF NOT EXISTS FOR (iv:IndicatorValue) ON (iv.indicator_name)
```
