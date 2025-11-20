# Instructions de configuration - RAG Communes Corses

## Prérequis

- Python 3.9 ou supérieur
- pip
- Environnement virtuel (recommandé)

## Installation dans un nouvel environnement

### 1. Créer un environnement virtuel

#### Sur Windows :
```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
venv\Scripts\activate
```

#### Sur Linux/Mac :
```bash
# Créer l'environnement
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

**Note :** L'installation peut prendre plusieurs minutes, surtout pour PyTorch et les modèles de transformers.

### 3. Télécharger les données NLTK (obligatoire)

Les données NLTK seront téléchargées automatiquement lors de la première exécution, mais vous pouvez les télécharger manuellement :

```python
import nltk

nltk.download('punkt_tab')
```

### 4. Préparer les fichiers de données

Assurez-vous d'avoir les fichiers suivants dans le répertoire :

- `data_comp_finalede2604_4_cleaned.csv` : Données principales des communes
- `sortie_questionnaire_traited.csv` : Données du questionnaire
- `communes_corse_wikipedia.csv` (optionnel) : Données Wikipedia scrappées

### 5. Configurer la clé OpenAI (optionnel)

Si vous souhaitez utiliser la génération de réponses avec GPT :

1. Créez un compte sur [platform.openai.com](https://platform.openai.com)
2. Générez une clé API
3. Créez un fichier `.env` avec votre clé :

```bash
OPENAI_API_KEY=votre_cle_api_ici
```

4. Modifiez le script pour charger la clé depuis l'environnement :

```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

5. Installez python-dotenv :
```bash
pip install python-dotenv
```

## Exécution du script

### Exécution complète

```bash
python rag_v1_2904.py
```

### Exécution avec scraping Wikipedia activé

Modifiez dans le fichier `rag_v1_2904.py` la ligne :

```python
scrape_wikipedia(DATA_CSV_PATH, enable_scraping=True)  # Mettre True
```

## Structure des fichiers générés

Après exécution, vous aurez :

```
fichiers_pour_rag/
├── rag_v1_2904.py              # Script principal
├── requirements.txt             # Dépendances
├── SETUP_INSTRUCTIONS.md        # Ce fichier
├── communes_text.txt            # Descriptions des communes (générées)
├── communes_corses_wiki.txt     # Textes Wikipedia (générés)
├── embeddings.pkl               # Cache des embeddings (généré)
├── chroma_txt/                  # Base de données vectorielle (générée)
├── rag_quanti/                  # Fichiers RAG quantitatifs (générés)
│   ├── Commune1.txt
│   ├── Commune2.txt
│   └── ...
├── rag_be/                      # Fichiers RAG bien-être (générés)
│   ├── Commune1.txtnltk.download('punkt')
│   ├── Commune2.txt
│   └── ...
├── df_mean_by_commune.csv       # Moyennes par commune (générées)
├── verbatims_by_commune.csv     # Verbatims par commune (générées)
└── dimension_counts.csv         # Comptages dimensions (générées)
```

## Utilisation du script

### 1. Recherche simple

Le script effectue automatiquement une recherche d'exemple. Pour modifier la question :

```python
question = "Votre question ici"
results = search_rag(collection, question, n_results=5)
```

### 2. Utilisation avec OpenAI

```python
from rag_v1_2904 import ask_rag_with_llm
import chromadb

# Connexion à la base
chroma_client = chromadb.PersistentClient(path="./chroma_txt")
collection = chroma_client.get_collection(name="communes_corses_txt")

# Poser une question
reponse = ask_rag_with_llm(
    collection,
    "Quelle est la qualité de vie à Ajaccio ?",
    openai_api_key="votre_cle_api",
    n_chunks=5,
    model_name="gpt-3.5-turbo"
)
print(reponse)
```

### 3. Interface Streamlit (optionnel)

Pour lancer une interface web :

1. Créez `app.py` :

```python
import streamlit as st
from rag_v1_2904 import ask_rag_with_llm
import chromadb

st.title("🐗 RAG Communes Corses")

chroma_client = chromadb.PersistentClient(path="./chroma_txt")
collection = chroma_client.get_collection(name="communes_corses_txt")

question = st.text_input("Posez votre question :")

if question:
    with st.spinner("Recherche..."):
        answer = ask_rag_with_llm(
            collection,
            question,
            openai_api_key="votre_cle",
            n_chunks=5
        )
    st.write(answer)
```

2. Lancez Streamlit :

```bash
streamlit run app.py
```

## Dépannage

### Erreur : "No module named 'torch'"

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Erreur : "numpy version incompatible"

```bash
pip install numpy==1.26.4 --force-reinstall
```

### Erreur : "ChromaDB connection failed"

Supprimez le dossier `chroma_txt/` et relancez le script.

### Erreur : "NLTK punkt not found"

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Mémoire insuffisante

Réduisez la taille des batches dans le script :

```python
batch_size = 64  # Au lieu de 128
```

## Commandes résumées

```bash
# 1. Créer et activer l'environnement
python -m venv venv
venv\Scripts\activate  # Windows
# ou : source venv/bin/activate  # Linux/Mac

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 3. Exécuter le script
python rag_v1_2904.py

# 4. Désactiver l'environnement quand terminé
deactivate
```

## Notes importantes

- **Première exécution** : Le téléchargement des modèles (transformers) peut prendre 10-20 minutes
- **Stockage** : Les modèles nécessitent ~2-3 GB d'espace disque
- **RAM** : Minimum 8 GB recommandé, 16 GB idéal
- **GPU** : Optionnel mais accélère grandement le traitement

## Support

Pour tout problème, vérifiez :
1. La version de Python (>= 3.9)
2. L'activation de l'environnement virtuel
3. Les logs d'erreur complets
4. La disponibilité des fichiers de données
