# API Chatbot RAG v3 - Documentation

API REST pour interroger le système RAG (Retrieval-Augmented Generation) enrichi par ontologie sur la qualité de vie en Corse.

## 📋 Table des matières

- [Installation](#installation)
- [Configuration](#configuration)
- [Démarrage rapide](#démarrage-rapide)
- [Endpoints](#endpoints)
- [Utilisation depuis un front-end](#utilisation-depuis-un-front-end)
- [Exemples](#exemples)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### 1. Installer les dépendances de base

```bash
# Si ce n'est pas déjà fait
pip install -r requirements.txt
```

### 2. Installer les dépendances API

```bash
pip install -r requirements_api.txt
```

Cela installe :
- `fastapi` - Framework web moderne
- `uvicorn` - Serveur ASGI performant
- `pydantic` - Validation de données
- `python-dotenv` - Gestion des variables d'environnement

---

## ⚙️ Configuration

### 1. Créer le fichier `.env`

```bash
# Copier le template
cp .env.example .env

# Éditer avec votre éditeur favori
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 2. Configurer votre clé API OpenAI

Éditer `.env` et remplacer :

```env
OPENAI_API_KEY=sk-votre-vraie-cle-api-ici
```

**Obtenir une clé API :** https://platform.openai.com/api-keys

### 3. Configuration optionnelle

Les autres paramètres ont des valeurs par défaut :

```env
# Chemins des données (optionnel)
# ONTOLOGY_PATH=ontology_be_2010.ttl
# CHROMA_PATH=./chroma_v2/
# COLLECTION_NAME=communes_corses_v2
# QUANT_DATA_PATH=df_mean_by_commune.csv

# Serveur (optionnel)
# API_HOST=0.0.0.0
# API_PORT=8000
# API_RELOAD=false
```

---

## 🏃 Démarrage rapide

### Lancer le serveur

```bash
python api_server.py
```

Vous devriez voir :

```
============================================================
DÉMARRAGE DU SERVEUR API
============================================================
Host: 0.0.0.0
Port: 8000
Documentation: http://localhost:8000/docs
============================================================

============================================================
INITIALISATION DU SYSTÈME RAG v3
============================================================
Initialisation de l'ontologie...
OK Ontologie initialisée
Chargement du cache depuis embeddings_v2.pkl...
OK 1234 documents chargés depuis le cache
Initialisation du retriever hybride depuis le cache...
OK Retriever hybride initialisé

============================================================
✓ SYSTÈME RAG v3 INITIALISÉ AVEC SUCCÈS
============================================================
```

### Accéder à la documentation interactive

Ouvrir dans votre navigateur : **http://localhost:8000/docs**

Vous aurez accès à l'interface Swagger UI pour tester l'API directement.

---

## 📡 Endpoints

### 1. `GET /` - Root

Informations de base sur l'API.

**Exemple :**
```bash
curl http://localhost:8000/
```

**Réponse :**
```json
{
  "message": "API Chatbot RAG v3 - Qualité de vie en Corse",
  "documentation": "/docs",
  "health_check": "/api/health",
  "query_endpoint": "/api/query"
}
```

---

### 2. `GET /api/health` - Health Check

Vérifie l'état du serveur et du système RAG.

**Exemple :**
```bash
curl http://localhost:8000/api/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "timestamp": "2025-11-15T10:30:00",
  "version": "1.0.0"
}
```

---

### 3. `POST /api/query` - Poser une question

Endpoint principal pour interroger le chatbot.

**Paramètres (body JSON) :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `question` | string | **requis** | Question à poser |
| `k` | integer | 5 | Nombre de documents à récupérer (1-20) |
| `use_reranking` | boolean | true | Utiliser le reranking |
| `include_quantitative` | boolean | true | Inclure les données quantitatives |
| `commune_filter` | string | null | Filtrer par commune |
| `use_ontology_enrichment` | boolean | true | Utiliser l'ontologie |

**Exemple avec curl :**

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelles sont les communes avec le meilleur bien-être ?",
    "k": 5,
    "use_reranking": true,
    "include_quantitative": true,
    "use_ontology_enrichment": true
  }'
```

**Réponse :**

```json
{
  "answer": "Selon les données disponibles, les communes de Corse qui présentent les meilleurs indicateurs de bien-être sont...",
  "sources": [
    {
      "content": "Extrait d'entretien pertinent...",
      "score": 0.85,
      "metadata": {
        "commune": "Ajaccio",
        "source": "entretien"
      }
    },
    {
      "content": "Autre extrait...",
      "score": 0.78,
      "metadata": {
        "commune": "Bastia",
        "source": "wiki"
      }
    }
  ],
  "metadata": {
    "k": 5,
    "use_reranking": true,
    "use_ontology": true,
    "include_quantitative": true,
    "commune_filter": null,
    "num_sources": 5
  },
  "timestamp": "2025-11-15T10:30:00"
}
```

---

## 🌐 Utilisation depuis un front-end

### JavaScript / React / Vue / Angular

```javascript
// Fonction pour interroger le chatbot
async function askChatbot(question) {
  try {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        k: 5,
        use_reranking: true,
        include_quantitative: true,
        use_ontology_enrichment: true
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('Erreur lors de la requête:', error);
    throw error;
  }
}

// Utilisation
askChatbot("Quelle commune a le meilleur bien-être ?")
  .then(data => {
    console.log("Réponse:", data.answer);
    console.log("Sources:", data.sources);
  })
  .catch(error => {
    console.error("Erreur:", error);
  });
```

### Exemple React avec useState

```jsx
import { useState } from 'react';

function Chatbot() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          k: 5,
          use_reranking: true,
          include_quantitative: true,
          use_ontology_enrichment: true
        })
      });

      const data = await response.json();
      setAnswer(data);

    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Posez votre question..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Chargement...' : 'Envoyer'}
        </button>
      </form>

      {answer && (
        <div>
          <h3>Réponse :</h3>
          <p>{answer.answer}</p>

          <h4>Sources :</h4>
          <ul>
            {answer.sources.map((source, idx) => (
              <li key={idx}>
                <strong>{source.metadata.commune}</strong> (score: {source.score.toFixed(2)})
                <p>{source.content.substring(0, 200)}...</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Chatbot;
```

### Python (requests)

```python
import requests

def ask_chatbot(question, k=5):
    url = "http://localhost:8000/api/query"

    payload = {
        "question": question,
        "k": k,
        "use_reranking": True,
        "include_quantitative": True,
        "use_ontology_enrichment": True
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()

# Utilisation
result = ask_chatbot("Quelle commune a le meilleur bien-être ?")
print("Réponse:", result['answer'])
print(f"Basée sur {len(result['sources'])} sources")
```

---

## 🧪 Exemples de questions

Voici des exemples de questions que vous pouvez poser :

### Questions générales
- "Quelles sont les communes avec le meilleur bien-être ?"
- "Quels sont les principaux facteurs de qualité de vie en Corse ?"
- "Comment les habitants perçoivent-ils leur qualité de vie ?"

### Questions spécifiques à une commune
```json
{
  "question": "Comment est la qualité de vie à Ajaccio ?",
  "commune_filter": "Ajaccio"
}
```

### Questions quantitatives
- "Quelles communes ont les meilleurs indicateurs économiques ?"
- "Compare les indicateurs de bien-être entre Ajaccio et Bastia"

### Questions ontologiques
- "Quelles sont les dimensions du bien-être territorial ?"
- "Comment sont perçues les relations sociales dans les communes ?"

---

## 🔧 Troubleshooting

### Erreur : `OPENAI_API_KEY non trouvée`

**Solution :** Créer un fichier `.env` avec votre clé API OpenAI.

```bash
cp .env.example .env
# Éditer .env et ajouter votre clé
```

---

### Erreur : `ModuleNotFoundError: No module named 'fastapi'`

**Solution :** Installer les dépendances API.

```bash
pip install -r requirements_api.txt
```

---

### Erreur : `Aucun cache trouvé`

**Solution :** Vous devez d'abord indexer les documents. Exécuter :

```bash
python index_rag_v2.py
```

---

### Erreur CORS depuis le navigateur

Si vous voyez une erreur CORS dans la console du navigateur :

```
Access to fetch at 'http://localhost:8000/api/query' from origin 'http://localhost:3000'
has been blocked by CORS policy
```

**Solution :** Le serveur est déjà configuré pour accepter toutes les origines en développement. Vérifiez que :
1. Le serveur est bien démarré
2. Vous utilisez le bon port (8000 par défaut)
3. Vous faites bien une requête POST avec `Content-Type: application/json`

---

### Le serveur est lent au démarrage

**Normal !** Le système charge :
- L'ontologie (fichier .ttl)
- La base ChromaDB (embeddings)
- Les modèles de ML (embeddings, reranking)

Attendez de voir :
```
✓ SYSTÈME RAG v3 INITIALISÉ AVEC SUCCÈS
```

---

### Port 8000 déjà utilisé

**Solution :** Changer le port dans `.env` :

```env
API_PORT=8001
```

Ou directement :

```bash
API_PORT=8001 python api_server.py
```

---

## 📊 Monitoring

### Logs des requêtes

Les requêtes sont automatiquement loguées dans la console :

```
[2025-11-15T10:30:00] Nouvelle requête: Quelles sont les communes avec le meilleur bien-être ?
[2025-11-15T10:30:05] Réponse générée avec 5 sources
```

### Health check

Pour vérifier que l'API fonctionne (par exemple depuis un monitoring) :

```bash
curl http://localhost:8000/api/health
```

Retourne `200 OK` si tout va bien.

---

## 🚀 Déploiement (pour plus tard)

Pour déployer en production, vous aurez besoin de :

1. **Serveur de production** (pas le serveur de dev)
   ```bash
   uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Reverse proxy** (Nginx, Traefik, etc.)

3. **HTTPS** (Let's Encrypt)

4. **Configuration CORS** adaptée (pas `allow_origins=["*"]`)

5. **Rate limiting** (pour éviter l'abus)

6. **Docker** (optionnel mais recommandé)

---

## 📝 Notes

- **Budget 0** : L'API utilise OpenAI (GPT-3.5-turbo) qui est payant. Surveillez votre consommation sur https://platform.openai.com/usage
- **Tests locaux uniquement** : CORS est configuré pour accepter toutes les origines (`*`). En production, spécifier les domaines autorisés.
- **Sécurité** : Ne jamais commiter le fichier `.env` sur Git. Il est déjà dans `.gitignore`.
- **Performance** : Le premier appel est lent (chargement des modèles). Les suivants sont plus rapides.

---

## 🆘 Support

Si vous rencontrez des problèmes :

1. Vérifier les logs du serveur (dans la console)
2. Tester avec `/api/health` pour voir si le système est bien initialisé
3. Utiliser l'interface Swagger (`/docs`) pour tester directement
4. Vérifier que ChromaDB est bien indexé (fichier `embeddings_v2.pkl` existe)

---

## 📄 Licence

Projet de recherche - Université de Corse

---

**Bon développement ! 🚀**
