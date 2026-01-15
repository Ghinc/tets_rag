# 🚀 Instructions pour tester le Graph-RAG

## Configuration : Neo4j SANS authentification

Vous avez **Neo4j Desktop avec `dbms.security.auth_enabled=false`**
✅ Le code a été adapté pour fonctionner sans credentials !

---

## 📋 Prérequis

### 1. Neo4j Desktop lancé
- ✅ Votre database est démarrée
- ✅ L'ontologie est importée dans Neo4j
- ✅ `dbms.security.auth_enabled=false` dans la config

### 2. Vérifier que Neo4j est accessible
Ouvrir http://localhost:7474 dans votre navigateur
→ Devrait s'ouvrir sans demander de mot de passe

---

## 🔧 Installation des dépendances

```bash
# Installer les packages Python nécessaires
pip install neo4j>=5.0.0
pip install chromadb>=0.4.0
pip install openai>=1.0.0
pip install sentence-transformers>=2.2.0
pip install pandas>=2.0.0
pip install python-dotenv>=1.0.0
pip install tabulate>=0.9.0
```

**OU** avec le fichier requirements :

```bash
pip install -r requirements_neo4j_existant.txt
```

---

## ⚙️ Configuration des variables d'environnement

### Sur Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="votre_clé_openai"
```

### Sur Linux/Mac
```bash
export OPENAI_API_KEY="votre_clé_openai"
```

**Note:** Pas besoin de `NEO4J_PASSWORD` car l'authentification est désactivée !

---

## ✅ Étape 1 : Tester la connexion Neo4j

```bash
python test_neo4j_connection.py
```

**Ce script va :**
1. Se connecter à Neo4j (sans authentification)
2. Vérifier si votre ontologie est présente
3. Afficher les statistiques du graphe
4. Tester les performances

**Résultat attendu :**
```
📡 Test de connexion à Neo4j...
   URI: bolt://localhost:7687
   User: neo4j
   Auth: Sans authentification

✅ Connexion réussie !

📊 STATISTIQUES DU GRAPHE
─────────────────────────────────
Total de nœuds: XXX
Total de relations: XXX
Nœuds d'ontologie (Concept/Dimension/Indicator): XXX

🎯 ÉTAT DE L'ONTOLOGIE
─────────────────────────────────
✅ Ontologie détectée (XXX nœuds)
```

---

## ✅ Étape 2 : Lancer la démo Graph-RAG

```bash
python demo_graphrag.py
```

**Ce qui se passe :**
1. Connexion à Neo4j (sans authentification)
2. Chargement de l'ontologie depuis le fichier .ttl
3. **Détection automatique** que l'ontologie est déjà dans Neo4j
4. Import des données de communes (si `df_mean_by_commune.csv` existe)
5. Lancement du mode interactif

**Résultat attendu :**
```
🔗 GRAPH-RAG DEMO 🔗

🔍 Vérification de l'environnement...
ℹ️  NEO4J_PASSWORD non définie (connexion sans authentification)
✅ OPENAI_API_KEY définie

🚀 Initialisation du pipeline Graph-RAG...

Connexion à Neo4j: bolt://localhost:7687
  Mode: Sans authentification
✓ Connexion Neo4j établie

Chargement de l'ontologie...
OK Ontologie chargee : XXX triples

✓ Ontologie déjà présente dans Neo4j (import skippé)

✓ Graph-RAG pipeline initialisé

Choisissez un mode:
  1. Démo automatique (3 questions prédéfinies)
  2. Mode interactif (posez vos propres questions)

Votre choix (1 ou 2):
```

---

## ✅ Étape 3 : Tester avec une question

### Mode automatique (option 1)

Le script pose 3 questions prédéfinies :
1. "Quelles sont les principales dimensions du bien-être territorial ?"
2. "Comment est la santé à Ajaccio ?"
3. "Quelles communes ont les meilleurs scores en éducation ?"

### Mode interactif (option 2)

Vous pouvez poser vos propres questions :
```
❓ Votre question: Quelles dimensions du bien-être sont liées à la santé ?

🔍 Recherche en cours...

💡 Réponse:
[La réponse générée par le LLM avec contexte du graphe]

📚 5 sources utilisées
```

---

## 🐛 Dépannage

### Erreur "Connection refused"
```bash
# Vérifier que Neo4j Desktop est lancé
# Ouvrir http://localhost:7474
# Démarrer votre database dans Neo4j Desktop
```

### Erreur "OPENAI_API_KEY not found"
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

### Erreur "No module named 'neo4j'"
```bash
pip install neo4j
```

### Erreur "chromadb not found" ou "chroma_v2 directory not found"
```bash
# Vous devez d'abord créer la base ChromaDB avec vos documents
# Lancez d'abord rag_v2_improved.py pour indexer vos documents
python rag_v2_improved.py
```

### L'ontologie n'est pas détectée
```cypher
# Dans Neo4j Browser (http://localhost:7474)
# Vérifier que l'ontologie est bien là :
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension', 'Indicator'])
RETURN count(n)
```

Si le résultat est 0, l'ontologie n'est pas importée. Le code l'importera automatiquement.

---

## 📊 Utilisation dans votre propre code

```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

# Initialiser (sans password = sans authentification)
rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password=None,  # ← Pas d'authentification
    openai_api_key="votre_clé"
)

# L'ontologie existante est automatiquement détectée !

# Importer les données de communes (si pas encore fait)
rag.import_commune_data("df_mean_by_commune.csv")

# Poser une question
response, results = rag.query(
    "Quelles dimensions du bien-être sont liées à la santé ?",
    use_graph=True  # Utiliser le graphe Neo4j
)

print(response)

# Fermer les connexions
rag.close()
```

---

## 🎯 Checklist de vérification

Avant de tester :

- [ ] Neo4j Desktop est lancé
- [ ] Database démarrée dans Neo4j Desktop
- [ ] http://localhost:7474 accessible (sans mot de passe)
- [ ] `dbms.security.auth_enabled=false` dans la config Neo4j
- [ ] Variable `OPENAI_API_KEY` définie
- [ ] Dépendances Python installées (`pip install -r requirements_neo4j_existant.txt`)
- [ ] (Optionnel) ChromaDB créée avec vos documents

---

## 📚 Fichiers de test

| Fichier | Description | Commande |
|---------|-------------|----------|
| `test_neo4j_connection.py` | Teste la connexion et affiche les stats | `python test_neo4j_connection.py` |
| `demo_graphrag.py` | Démo interactive complète | `python demo_graphrag.py` |
| `rag_v5_graphrag_neo4j.py` | Code principal (à importer dans vos scripts) | - |

---

## 💡 Exemple de session complète

```bash
# 1. Définir la clé OpenAI
export OPENAI_API_KEY="sk-..."

# 2. Tester la connexion
python test_neo4j_connection.py
# → ✅ Connexion OK
# → ✅ Ontologie détectée (120 nœuds)

# 3. Lancer la démo
python demo_graphrag.py
# → Choisir mode 2 (interactif)

# 4. Poser des questions
❓ Votre question: Quelles sont les dimensions du bien-être ?
💡 Réponse: Le bien-être territorial comprend plusieurs dimensions...

❓ Votre question: Comment est la santé à Ajaccio ?
💡 Réponse: À Ajaccio, la dimension santé montre...

# 5. Quitter
❓ Votre question: quit
👋 Au revoir !
```

---

## 🎉 C'est prêt !

Vous pouvez maintenant :
✅ Utiliser le Graph-RAG avec votre Neo4j existant
✅ Exploiter votre ontologie déjà importée
✅ Poser des questions qui utilisent le graphe de connaissances
✅ Combiner retrieval vectoriel (ChromaDB) + graphe (Neo4j)

**Bon test ! 🚀**
