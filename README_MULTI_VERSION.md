# API Multi-Version - Guide d'utilisation

Ce guide explique comment utiliser le système multi-version qui permet de basculer facilement entre v1, v2 et v3 du RAG.

---

## 🚀 Démarrage rapide

### 1. Lancer le serveur multi-version

```bash
python api_server_multi_version.py
```

Le serveur initialise automatiquement **toutes les versions disponibles** (v1, v2, v3).

```
============================================================
INITIALISATION MULTI-VERSION RAG
============================================================

[1/3] Initialisation RAG v1...
OK RAG v1 initialisé

[2/3] Initialisation RAG v2...
OK RAG v2 initialisé

[3/3] Initialisation RAG v3...
OK RAG v3 initialisé

============================================================
SYSTEMES RAG DISPONIBLES: v1, v2, v3
============================================================
```

### 2. Ouvrir l'interface graphique

Double-cliquez sur **[example_frontend_multi_version.html](example_frontend_multi_version.html)**

Ou ouvrez-le dans votre navigateur.

### 3. Choisir une version et poser des questions

L'interface affiche les 3 versions avec leur statut :
- ✅ **v1** - RAG Basique (Disponible)
- ✅ **v2** - RAG Amélioré (Disponible)
- ✅ **v3** - RAG avec Ontologie (Disponible)

Cliquez sur une version, puis posez votre question !

---

## 📡 Endpoints API

### 1. `GET /api/versions` - Liste des versions

```bash
curl http://localhost:8000/api/versions
```

**Réponse :**
```json
[
  {
    "version": "v1",
    "name": "RAG Basique",
    "description": "Retrieval vectoriel simple avec génération LLM",
    "available": true,
    "features": [
      "Retrieval vectoriel (e5-base-v2)",
      "Génération avec GPT-3.5-turbo",
      "Simple et rapide"
    ]
  },
  {
    "version": "v2",
    "name": "RAG Amélioré",
    "description": "Hybrid retrieval (BM25 + Vector) avec reranking",
    "available": true,
    "features": [
      "Hybrid retrieval (BM25 + Vector)",
      "Reranking avec cross-encoder",
      "Données quantitatives",
      "Meilleure précision"
    ]
  },
  {
    "version": "v3",
    "name": "RAG avec Ontologie",
    "description": "v2 + enrichissement sémantique via ontologie",
    "available": true,
    "features": [
      "Toutes les fonctionnalités v2",
      "Enrichissement de requête via ontologie",
      "Compréhension sémantique avancée",
      "Meilleure couverture thématique"
    ]
  }
]
```

---

### 2. `POST /api/query` - Poser une question

**Avec v1 :**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les facteurs de bien-être ?",
    "rag_version": "v1",
    "k": 5
  }'
```

**Avec v2 :**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les facteurs de bien-être ?",
    "rag_version": "v2",
    "k": 5,
    "use_reranking": true,
    "include_quantitative": true
  }'
```

**Avec v3 :**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les facteurs de bien-être ?",
    "rag_version": "v3",
    "k": 5,
    "use_reranking": true,
    "include_quantitative": true,
    "use_ontology_enrichment": true
  }'
```

**Réponse (toutes versions) :**
```json
{
  "answer": "Les facteurs de bien-être sont...",
  "sources": [...],
  "metadata": {
    "k": 5,
    "use_reranking": true,
    "use_ontology": true,
    "num_sources": 5
  },
  "rag_version_used": "v3",
  "timestamp": "2025-11-15T11:20:00"
}
```

---

### 3. `GET /api/health` - Vérifier l'état

```bash
curl http://localhost:8000/api/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "rag_v1_initialized": true,
  "rag_v2_initialized": true,
  "rag_v3_initialized": true,
  "timestamp": "2025-11-15T11:20:00",
  "version": "2.0.0"
}
```

---

## 🔍 Comparaison des versions

### Performance et caractéristiques

| Critère | v1 | v2 | v3 |
|---------|----|----|---- |
| **Vitesse** | ⚡⚡⚡ Très rapide | ⚡⚡ Rapide | ⚡ Moyenne |
| **Précision** | ⭐⭐ Basique | ⭐⭐⭐ Bonne | ⭐⭐⭐⭐ Excellente |
| **Retrieval** | Vector seul | Hybrid (BM25+Vector) | Hybrid (BM25+Vector) |
| **Reranking** | ❌ | ✅ Cross-encoder | ✅ Cross-encoder |
| **Données quanti** | ❌ | ✅ | ✅ |
| **Ontologie** | ❌ | ❌ | ✅ |
| **Collection** | `chroma_txt/` | `chroma_v2/` | `chroma_v2/` |

### Quand utiliser quelle version ?

**v1 - RAG Basique**
- ✅ Rapidité primordiale
- ✅ Questions simples
- ✅ Prototype/démo
- ❌ Précision critique

**v2 - RAG Amélioré**
- ✅ Bon compromis vitesse/précision
- ✅ Questions complexes
- ✅ Besoin de données quantitatives
- ✅ Production (recommandé)

**v3 - RAG avec Ontologie**
- ✅ Précision maximale
- ✅ Compréhension sémantique
- ✅ Questions conceptuelles
- ✅ Recherche académique
- ❌ Contrainte de vitesse forte

---

## 🧪 Exemples de tests comparatifs

### Question simple

```bash
# Tester avec les 3 versions
for version in v1 v2 v3; do
  echo "=== Test $version ==="
  curl -s -X POST "http://localhost:8000/api/query" \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"Quelle commune a le meilleur bien-être ?\",\"rag_version\":\"$version\",\"k\":3}" \
    | python -c "import sys,json; d=json.load(sys.stdin); print(f'Réponse: {d[\"answer\"][:100]}...')"
  echo ""
done
```

### Depuis JavaScript

```javascript
async function compareVersions(question) {
  const versions = ['v1', 'v2', 'v3'];
  const results = {};

  for (const version of versions) {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        rag_version: version,
        k: 5
      })
    });

    results[version] = await response.json();
  }

  console.table({
    v1: { answer: results.v1.answer.substring(0, 50), sources: results.v1.sources.length },
    v2: { answer: results.v2.answer.substring(0, 50), sources: results.v2.sources.length },
    v3: { answer: results.v3.answer.substring(0, 50), sources: results.v3.sources.length }
  });
}

// Utilisation
compareVersions("Quels sont les facteurs de bien-être ?");
```

---

## 🔧 Configuration

### Désactiver certaines versions

Si vous ne voulez charger que certaines versions, modifiez `initialize_all_rags()` dans [api_server_multi_version.py](api_server_multi_version.py) :

```python
# Exemple : charger seulement v2 et v3
def initialize_all_rags():
    # ... code existant ...

    # Commenter pour désactiver v1
    # rag_pipelines["v1"] = BasicRAGPipeline(...)
    rag_pipelines["v1"] = None  # Désactivé

    # v2 et v3 restent actifs
    rag_pipelines["v2"] = ImprovedRAGPipeline(...)
    rag_pipelines["v3"] = RAGPipelineWithOntology(...)
```

---

## 📊 Interface graphique

L'interface [example_frontend_multi_version.html](example_frontend_multi_version.html) offre :

- ✅ **Sélection visuelle** des versions avec badges de statut
- ✅ **Paramètres configurables** (k, reranking, ontologie, etc.)
- ✅ **Affichage des sources** avec scores de pertinence
- ✅ **Badge de version** sur chaque réponse
- ✅ **Gestion d'erreurs** claire

### Fonctionnalités

1. **Détection automatique** des versions disponibles
2. **Basculement instantané** entre versions
3. **Paramètres adaptatifs** (reranking grisé pour v1, ontologie pour v3 uniquement)
4. **Historique visuel** avec indication de la version utilisée

---

## 🆚 Différences avec l'API simple version

| Fonctionnalité | API simple (`api_server.py`) | API multi-version (`api_server_multi_version.py`) |
|----------------|------------------------------|--------------------------------------------------|
| Versions chargées | 1 seule (hardcodée) | Toutes (v1, v2, v3) |
| Changement de version | Modification du code | Paramètre dans la requête |
| Mémoire utilisée | ~500 MB | ~1.5 GB (3 modèles chargés) |
| Temps démarrage | ~15 sec | ~45 sec (3 initialisations) |
| Utilisation | Production (1 version) | Tests et comparaisons |

---

## ⚠️ Notes importantes

1. **Mémoire** : Le serveur multi-version charge **tous les modèles en même temps**. Nécessite ~2 GB de RAM.

2. **Collections** : v1 utilise `chroma_txt/`, v2 et v3 utilisent `chroma_v2/`. Assurez-vous que les deux existent.

3. **Versions manquantes** : Si une collection n'existe pas, la version correspondante sera désactivée automatiquement.

4. **API key** : La même clé OpenAI est utilisée pour toutes les versions.

---

## 🚀 Production

Pour la production, utilisez **`api_server.py`** (version unique) plutôt que `api_server_multi_version.py` :

- Moins de mémoire
- Plus rapide au démarrage
- Performance optimale

Le système multi-version est idéal pour :
- 🧪 Tests comparatifs
- 📊 Benchmarking
- 🎓 Démonstrations
- 🔬 Recherche

---

**Bon testing ! 🎉**
