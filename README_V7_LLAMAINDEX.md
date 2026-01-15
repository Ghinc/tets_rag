# RAG v7 - LlamaIndex Pipeline

## Vue d'ensemble

**v7** est une implémentation du RAG utilisant **LlamaIndex** comme framework d'orchestration, offrant :

- 🎯 **Router automatique** : Choix intelligent entre retrieval vectoriel et graphe
- 🧩 **Décomposition de questions** : Questions complexes → sous-questions
- 🔀 **Hybrid retrieval** : Fusion pondérée vector + graph (comme v6)
- 📊 **Observabilité native** : Tracing et métriques automatiques
- 🚀 **Code simplifié** : ~200 lignes vs 500+ pour v2-v6

## Architecture

```
┌────────────────────────────────────────────────┐
│         RAG v7 - LlamaIndex Pipeline            │
├────────────────────────────────────────────────┤
│                                                 │
│  Indexes:                                      │
│  ├── VectorStoreIndex (ChromaDB - 5815 docs)  │
│  └── PropertyGraphIndex (Neo4j - 116 nodes)   │
│                                                 │
│  Query Engines:                                │
│  ├── RouterQueryEngine (auto vector/graph)    │
│  ├── SubQuestionQueryEngine (décomposition)   │
│  └── HybridRetriever (fusion 0.4/0.6)        │
│                                                 │
└────────────────────────────────────────────────┘
```

## Installation

### 1. Installer les dépendances LlamaIndex

```bash
pip install -r requirements_llamaindex.txt
```

### 2. Vérifier les prérequis

- ✅ ChromaDB v2 avec 5815 documents (déjà prêt)
- ✅ Neo4j avec ontologie BE-2010 (déjà prêt)
- ✅ OPENAI_API_KEY dans `.env`

## Utilisation

### Test rapide

```bash
python test_v7_llamaindex.py
```

### Utilisation programmatique

```python
from rag_v7_llamaindex import LlamaIndexRAGPipeline

# Initialiser
rag = LlamaIndexRAGPipeline(
    openai_api_key="your-key",
    chroma_path="./chroma_v2",
    collection_name="communes_corses_v2"
)

# Mode 1: Router (automatique)
response, sources = rag.query(
    "Quels sont les scores à Ajaccio ?",
    mode="router"
)
# → Route automatiquement vers vector retrieval

# Mode 2: Sub-question (questions complexes)
response, sources = rag.query(
    "Compare santé et logement à Bastia",
    mode="sub_question"
)
# → Décompose en 2 sous-questions

# Mode 3: Hybrid (fusion vector+graph)
response, sources = rag.query(
    "Qu'est-ce que la santé et ses scores à Ajaccio ?",
    mode="hybrid"
)
# → Combine vector (0.4) + graph (0.6)
```

## Modes de requête

### 1. Router Mode (défaut)

**Cas d'usage** : Laisser le LLM choisir la meilleure stratégie

```python
response, sources = rag.query(
    "Qu'est-ce que la dimension santé ?",
    mode="router"
)
# LLM détecte : question conceptuelle → GRAPH
```

**Avantages** :
- Automatique, pas besoin de décider
- Utilise LLM pour classifier la question
- Optimal pour questions simples

**Inconvénients** :
- Coût LLM supplémentaire
- Peut se tromper de route

### 2. Sub-Question Mode

**Cas d'usage** : Questions complexes nécessitant plusieurs étapes

```python
response, sources = rag.query(
    "Compare la santé, le logement et l'éducation entre Ajaccio et Bastia",
    mode="sub_question"
)
# Décompose en 6 sous-questions :
# 1. Santé à Ajaccio ?
# 2. Santé à Bastia ?
# 3. Logement à Ajaccio ?
# ... etc
```

**Avantages** :
- Gère questions multi-facettes
- Trace chaque sous-question
- Réponses plus complètes

**Inconvénients** :
- Plus lent (plusieurs requêtes)
- Plus coûteux en tokens

### 3. Hybrid Mode

**Cas d'usage** : Questions mixtes (concept + données)

```python
response, sources = rag.query(
    "Explique la dimension santé et donne les scores d'Ajaccio",
    mode="hybrid"
)
# Combine :
# - Graph : définition de "dimension santé"
# - Vector : scores d'Ajaccio
# Fusion : graph (0.6) + vector (0.4)
```

**Avantages** :
- Combine le meilleur des deux mondes
- Pondération comme v6
- Une seule requête

**Inconvénients** :
- Peut ramener sources non pertinentes
- Pas de routing intelligent

## Comparaison avec v2-v6

| Feature | v2-v6 | v7 (LlamaIndex) |
|---------|-------|-----------------|
| **Code** | 500+ lignes/version | ~200 lignes total |
| **Routing** | Manuel (if/else) | Automatique (LLM) |
| **Multi-index** | Custom fusion | `RouterQueryEngine` |
| **Citations** | Tracking manuel | Natif (`source_nodes`) |
| **Décomposition** | v4 uniquement | `SubQuestionQueryEngine` |
| **Observabilité** | Prints | Callbacks + tracing |
| **Extensibilité** | Difficile | Facile (agents, etc.) |

## Performances attendues

### Vitesse
- **Router mode** : ~3-5s (LLM routing + retrieval + génération)
- **Sub-question mode** : ~10-20s (multiple retrievals)
- **Hybrid mode** : ~2-4s (retrieval parallèle + génération)

### Qualité
- **Router** : Excellente si routing correct, sinon peut manquer info
- **Sub-question** : Très bonne pour questions complexes
- **Hybrid** : Bonne couverture, peut être moins précis

## Intégration dans le serveur API

Pour ajouter v7 au serveur multi-version :

```python
# Dans api_server_multi_version.py

from rag_v7_llamaindex import LlamaIndexRAGPipeline

# Initialisation
rag_v7 = LlamaIndexRAGPipeline(
    openai_api_key=OPENAI_API_KEY,
    chroma_path="./chroma_v2",
    collection_name="communes_corses_v2"
)

# Endpoint avec mode
@app.post("/api/query")
async def query(request: QueryRequest):
    if request.version == "v7":
        response, sources = rag_v7.query(
            question=request.question,
            mode=request.mode or "router",  # Défaut: router
            commune_filter=request.commune_filter,
            k=request.k or 5
        )
        # ... format response
```

## Évolutions futures

### Court terme
- [ ] Intégrer dans `api_server_multi_version.py`
- [ ] Ajouter paramètre `mode` dans frontend
- [ ] Benchmarker vs v2-v6

### Moyen terme
- [ ] **Streaming** : Réponses en temps réel avec `stream_chat()`
- [ ] **Agents** : `ReActAgent` pour raisonnement multi-étapes
- [ ] **Fine-tuning** : Embeddings fine-tunés sur domaine communes
- [ ] **Évaluation** : Metrics (Faithfulness, Relevance, etc.)

### Long terme
- [ ] **Multimodal** : Intégrer images/cartes des communes
- [ ] **Knowledge graphs enrichis** : Graphe communes + relations
- [ ] **Personnalisation** : User preferences, history

## Troubleshooting

### Erreur : "Collection does not exist"
```bash
# ChromaDB v2 n'existe pas encore
python import_all_sources.py
```

### Erreur : "Neo4j connection failed"
```bash
# Démarrer Neo4j
start http://localhost:7474
```

### Erreur : "OPENAI_API_KEY missing"
```bash
# Vérifier .env
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows
```

### LlamaIndex trop verbeux
```python
import logging
logging.getLogger("llama_index").setLevel(logging.WARNING)
```

## Références

- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [VectorStoreIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index.html)
- [PropertyGraphIndex](https://docs.llamaindex.ai/en/stable/examples/property_graph/)
- [Query Engines](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/)

## Auteur

Implémenté par Claude Code - Janvier 2026
