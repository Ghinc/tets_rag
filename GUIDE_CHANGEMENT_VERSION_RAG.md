# Guide : Changer la version du RAG dans l'API

Ce guide explique comment basculer entre les différentes versions du RAG (v1, v2, v3) dans l'API.

---

## 📋 Versions disponibles

| Version | Fichier | Classe | Fonctionnalités |
|---------|---------|--------|-----------------|
| **v1** | `rag_v1_class.py` | `BasicRAGPipeline` | Simple retrieval vectoriel |
| **v2** | `rag_v2_improved.py` | `ImprovedRAGPipeline` | Hybrid retrieval (BM25 + Vector) + Reranking |
| **v3** | `rag_v3_ontology.py` | `RAGPipelineWithOntology` | v2 + Enrichissement ontologique |

---

## 🔄 Comment changer de version

### Option 1 : Passer à RAG v1 (simple)

#### Étape 1 : Modifier les imports dans `api_server.py`

**Ligne 27** - Remplacer :
```python
from rag_v3_ontology import RAGPipelineWithOntology, RetrievalResult
```

Par :
```python
from rag_v1_class import BasicRAGPipeline, RetrievalResult
```

#### Étape 2 : Modifier l'initialisation dans `initialize_rag()`

**Lignes 159-168** - Remplacer :
```python
rag_pipeline = RAGPipelineWithOntology(
    openai_api_key=openai_api_key,
    ontology_path=ontology_path,
    chroma_path=chroma_path,
    collection_name=collection_name,
    quant_data_path=quant_data_path,
    llm_model="gpt-3.5-turbo",
    embedding_model="intfloat/e5-base-v2",
    reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
)
```

Par :
```python
rag_pipeline = BasicRAGPipeline(
    openai_api_key=openai_api_key,
    chroma_path="./chroma_txt/",  # ← Collection v1
    collection_name="communes_corses_txt",  # ← Nom v1
    llm_model="gpt-3.5-turbo",
    embedding_model="intfloat/e5-base-v2"
    # Retirer : ontology_path, quant_data_path, reranker_model
)
```

#### Étape 3 : Modifier l'appel à `query()`

**Lignes 242-249** - Remplacer :
```python
answer, retrieval_results = rag_pipeline.query(
    question=request.question,
    k=request.k,
    use_reranking=request.use_reranking,
    include_quantitative=request.include_quantitative,
    commune_filter=request.commune_filter,
    use_ontology_enrichment=request.use_ontology_enrichment
)
```

Par :
```python
# v1 accepte tous les paramètres mais n'utilise que question et k
answer, retrieval_results = rag_pipeline.query(
    question=request.question,
    k=request.k
    # Les autres paramètres sont ignorés en v1
)
```

#### Étape 4 : Optionnel - Mettre à jour les messages

**Ligne 171** - Remplacer :
```python
print("OK SYSTEME RAG v3 INITIALISE AVEC SUCCES")
```

Par :
```python
print("OK SYSTEME RAG v1 INITIALISE AVEC SUCCES")
```

**Ligne 114** - Remplacer le titre de l'API :
```python
app = FastAPI(
    title="API Chatbot RAG v1 - Qualité de vie en Corse",  # ← v1
    description="API REST pour interroger le système RAG basique sur la qualité de vie en Corse",
    ...
)
```

---

### Option 2 : Passer à RAG v2

#### Étape 1 : Modifier l'import

**Ligne 27** :
```python
from rag_v2_improved import ImprovedRAGPipeline, RetrievalResult
```

#### Étape 2 : Modifier l'initialisation

**Lignes 159-168** :
```python
rag_pipeline = ImprovedRAGPipeline(
    openai_api_key=openai_api_key,
    chroma_path=chroma_path,
    collection_name=collection_name,
    quant_data_path=quant_data_path,
    llm_model="gpt-3.5-turbo",
    embedding_model="intfloat/e5-base-v2",
    reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
    # Retirer : ontology_path
)
```

#### Étape 3 : Modifier l'appel à `query()`

**Lignes 242-249** :
```python
answer, retrieval_results = rag_pipeline.query(
    question=request.question,
    k=request.k,
    use_reranking=request.use_reranking,
    include_quantitative=request.include_quantitative,
    commune_filter=request.commune_filter
    # Retirer : use_ontology_enrichment
)
```

---

### Option 3 : Revenir à RAG v3 (actuel)

Aucun changement nécessaire, c'est déjà configuré !

---

## ⚠️ Points d'attention

### 1. **Collections ChromaDB différentes**

Chaque version utilise potentiellement une collection différente :

- **v1** : `communes_corses_txt` dans `./chroma_txt/`
- **v2** : `communes_corses_v2` dans `./chroma_v2/`
- **v3** : `communes_corses_v2` dans `./chroma_v2/` (même que v2)

Assurez-vous que la collection existe avant de démarrer l'API !

### 2. **Modèles d'embeddings**

Toutes les versions utilisent `intfloat/e5-base-v2` pour votre setup actuel.

Si vous avez indexé avec un modèle différent, modifiez le paramètre `embedding_model`.

### 3. **Paramètres de l'API**

L'API expose tous les paramètres (reranking, ontology, etc.) même si la version ne les utilise pas.

En v1, ces paramètres sont simplement ignorés (grâce à `**kwargs`).

### 4. **Fichiers nécessaires**

| Version | Fichiers requis |
|---------|-----------------|
| **v1** | `chroma_txt/` avec collection indexée |
| **v2** | `chroma_v2/` + `df_mean_by_commune.csv` |
| **v3** | `chroma_v2/` + `df_mean_by_commune.csv` + `ontology_be_2010.ttl` |

---

## 🧪 Tester après changement

Après avoir modifié `api_server.py` :

```bash
# 1. Redémarrer le serveur
python api_server.py

# 2. Tester le health check
curl http://localhost:8000/api/health

# 3. Tester une requête
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelle commune a le meilleur bien-être ?","k":3}'
```

---

## 📝 Exemple complet : Basculer vers v1

### Modifications dans `api_server.py`

```python
# ============= LIGNE 27 =============
from rag_v1_class import BasicRAGPipeline, RetrievalResult

# ============= LIGNES 159-168 =============
rag_pipeline = BasicRAGPipeline(
    openai_api_key=openai_api_key,
    chroma_path="./chroma_txt/",
    collection_name="communes_corses_txt",
    llm_model="gpt-3.5-turbo",
    embedding_model="intfloat/e5-base-v2"
)

# ============= LIGNES 242-249 =============
answer, retrieval_results = rag_pipeline.query(
    question=request.question,
    k=request.k
)
```

C'est tout ! 🎉

---

## 💡 Alternative : Configuration via `.env`

Si vous voulez changer de version sans modifier le code, vous pouvez créer un système de configuration.

Ajoutez dans `.env` :
```env
RAG_VERSION=v1  # v1, v2, ou v3
```

Et modifiez `api_server.py` pour charger dynamiquement la bonne classe.

Voulez-vous que je crée ce système ?

---

## 🆘 Problèmes courants

### "Collection not found"

➜ La collection n'existe pas. Vérifiez que vous avez bien indexé les données avec la bonne version.

**Solution v1** :
```bash
python rag_v1_2904.py
```

**Solution v2** :
```bash
python index_rag_v2.py
```

### "Dimension mismatch"

➜ Le modèle d'embeddings ne correspond pas à celui utilisé lors de l'indexation.

**Solution** : Vérifier `embedding_model` dans `api_server.py` et dans le script d'indexation.

### "Ontology file not found" (v3 uniquement)

➜ Le fichier `ontology_be_2010.ttl` est manquant.

**Solution** : Placer le fichier à la racine du projet.

---

**Bon développement ! 🚀**
