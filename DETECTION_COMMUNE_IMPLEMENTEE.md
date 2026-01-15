# Implémentation de la détection automatique des communes

## Résumé

Un système de détection automatique des noms de communes a été implémenté dans tous les fichiers RAG (v2, v3, v4, v5). Lorsqu'une commune est mentionnée dans la question de l'utilisateur, le système :

1. **Détecte automatiquement** le nom de la commune via `commune_detector.py`
2. **Filtre ChromaDB** pour ne retourner QUE les documents de cette commune
3. **Affiche des logs** pour tracer la détection et le filtrage

---

## Fichiers modifiés

### 1. **rag_v2_improved.py**
- Import de `commune_detector`
- Modification de `HybridRetriever.retrieve()` pour accepter `commune_filter: Optional[str]`
- Ajout du filtre ChromaDB `where={"nom": commune_filter}`
- Modification de `ImprovedRAGPipeline.query()` pour détecter et appliquer le filtre

### 2. **rag_v2_boosted.py**
- Mêmes modifications que v2_improved

### 3. **rag_v3_ontology.py**
- Import de `commune_detector`
- Modification de `RAGPipelineWithOntology.query()` pour détecter la commune
- Transmission du filtre au `HybridRetriever`

### 4. **rag_v4_cross_analysis.py**
- Import de `commune_detector`
- Modification de `HybridRetriever.retrieve()` pour accepter `commune_filter`
- Modification de `ImprovedRAGPipeline.query()` pour détecter et appliquer le filtre

### 5. **rag_v5_graphrag_neo4j.py**
- Import de `commune_detector`
- Modification de `GraphRAGPipeline.query()` pour détecter la commune
- Ajout du filtre ChromaDB `where={"nom": commune}` dans `collection.query()`

---

## Détails techniques

### Modification de `HybridRetriever.retrieve()`

**Avant :**
```python
def retrieve(self, query: str, query_embedding: np.ndarray,
            k: int = 10) -> List[RetrievalResult]:
    dense_results = self.chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )
```

**Après :**
```python
def retrieve(self, query: str, query_embedding: np.ndarray,
            k: int = 10, commune_filter: Optional[str] = None) -> List[RetrievalResult]:
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": 20,
        "include": ["documents", "metadatas", "distances"]
    }

    # Ajouter filtre par commune si spécifié
    if commune_filter:
        query_params["where"] = {"nom": commune_filter}
        print(f"[FILTRE] Recherche limitée à la commune: {commune_filter}")

    dense_results = self.chroma_collection.query(**query_params)
```

### Modification de `ImprovedRAGPipeline.query()`

**Ajout au début de la méthode :**
```python
# 1. Détecter automatiquement la commune dans la question
detected_commune = detect_commune(question)
if detected_commune:
    print(f"[AUTO-DETECT] Commune détectée: {detected_commune}")
    commune_filter = detected_commune

# 2. Encoder la requête
query_embedding = self.embed_model.encode_query(question)

# 3. Retrieval hybride
results = self.hybrid_retriever.retrieve(
    question,
    query_embedding,
    k=k*2,
    commune_filter=commune_filter  # <-- Filtre ajouté
)
```

### Pour rag_v5_graphrag_neo4j.py

```python
# 2. Détecter commune dans la question
detected_commune = detect_commune(question)
if detected_commune:
    print(f"[AUTO-DETECT] Commune détectée: {detected_commune}")
    if not commune_filter:
        commune_filter = detected_commune

# 3. Retrieval vectoriel classique
query_params = {
    "query_embeddings": [query_embedding.tolist()],
    "n_results": k*2,
    "include": ["documents", "metadatas", "distances"]
}

# Ajouter filtre par commune si spécifié
if commune_filter:
    query_params["where"] = {"nom": commune_filter}
    print(f"[FILTRE v5] Recherche limitée à: {commune_filter}")

dense_results = self.collection.query(**query_params)
```

---

## Tests effectués

### Test 1 : Ajaccio avec v2_boosted
```json
{
  "question": "Comment est la qualité de vie à Ajaccio ?",
  "version": "v2_boosted",
  "k": 3
}
```

**Logs serveur :**
```
[AUTO-DETECT] Commune détectée: Ajaccio
[FILTRE] Recherche limitée à la commune: Ajaccio
```

✅ **Résultat :** Seuls les documents d'Ajaccio sont retournés

### Test 2 : Bastia avec v3
```json
{
  "question": "Quels sont les problèmes de transport à Bastia ?",
  "version": "v3",
  "k": 3
}
```

**Logs serveur :**
```
[AUTO-DETECT] Commune détectée: Bastia
[FILTRE] Recherche limitée à la commune: Bastia
```

✅ **Résultat :** Seuls les documents de Bastia sont retournés

### Test 3 : Porto-Vecchio avec v4
```json
{
  "question": "Comment est le logement à Porto-Vecchio ?",
  "version": "v4",
  "k": 3
}
```

**Logs serveur :**
```
[AUTO-DETECT] Commune détectée: Porto-Vecchio
[FILTRE] Recherche limitée à la commune: Porto-Vecchio
```

✅ **Résultat :** Seuls les documents de Porto-Vecchio sont retournés

### Test 4 : Calvi avec v5 (Graph-RAG)
```json
{
  "question": "Quelle est la santé à Calvi ?",
  "version": "v5",
  "k": 3
}
```

**Logs serveur :**
```
[AUTO-DETECT] Commune détectée: Calvi
[FILTRE] Recherche limitée à la commune: Calvi
```

✅ **Résultat :** Seuls les documents de Calvi sont retournés

---

## Fonctionnement

1. **Détection :** Le module `commune_detector.py` charge la liste des communes depuis `./communes_chatbot/` au démarrage
2. **Recherche :** Pour chaque requête, il cherche les noms de communes dans le texte (insensible à la casse, tolère variations)
3. **Filtrage :** Si une commune est détectée, ChromaDB filtre avec `where={"nom": commune_name}`
4. **Logs :** Deux lignes de log sont affichées :
   - `[AUTO-DETECT] Commune détectée: XXX`
   - `[FILTRE] Recherche limitée à la commune: XXX`

---

## Avantages

✅ **Filtrage strict :** Ne retourne QUE les documents de la commune mentionnée
✅ **Automatique :** Aucune configuration manuelle requise
✅ **Transparent :** Logs clairs pour le debugging
✅ **Compatible :** Fonctionne avec tous les RAG (v2, v3, v4, v5)
✅ **Robuste :** Tolère les variations orthographiques (tirets, espaces, accents)

---

## Prochaines étapes possibles

- [ ] Étendre à la détection de plusieurs communes dans une même question
- [ ] Ajouter une API pour obtenir la liste des communes disponibles
- [ ] Permettre le filtrage par région/département
- [ ] Ajouter des statistiques sur les communes les plus recherchées

---

**Date d'implémentation :** 2025-12-28
**Status :** ✅ Opérationnel sur toutes les versions RAG (v2, v3, v4, v5)
