# 📋 Information sur les fichiers ignorés par Git

Ce document explique ce qui est exclu du dépôt Git via le fichier `.gitignore`.

## 🚫 Fichiers et dossiers ignorés

### 📊 Données volumineuses (> 1MB)
- **Fichiers Excel** : `*.xlsx`, `*.xls`
- **Fichiers GeoJSON** : `*.geojson` (très volumineux, ~3-5MB)
- **Fichiers Shapefile** : `*.shp`, `*.shx`, `*.dbf`, `*.prj`, etc.
- **Fichiers CSV** : `*.csv` (données brutes)

### 📁 Dossiers exclus
- `Code/OUTPUT/` : Tous les résultats générés (LISA, CAH, exports)
- `Code/Archive/` : Anciennes versions du code
- `Code/Python/Archive_anciennes_normalisations/` : Anciennes normalisations
- `Code/Python/Excel_doublons/` : Fichiers Excel en double
- `Code/WEB/archive/` : Anciennes versions web

### 🖼️ Fichiers générés
- **Images** : `*.png`, `*.jpg`, `*.pdf` (sauf fichiers web essentiels)
  - Exception : `cah_3_clusters_ecarts.png`, `cah_5_clusters_ecarts.png`
- **Graphiques LISA et CAH** : Tous les graphiques dans `OUTPUT/`

### 🐍 Python et environnements
- `__pycache__/`, `*.pyc` : Cache Python
- `venv/`, `.venv/`, `env/` : Environnements virtuels
- `.ipynb_checkpoints/` : Checkpoints Jupyter

### 💻 IDE et éditeurs
- `.vscode/`, `.idea/` : Configuration des éditeurs
- `*.sublime-*` : Sublime Text

### 🖥️ Système d'exploitation
- Windows : `Thumbs.db`, `Desktop.ini`
- macOS : `.DS_Store`
- Linux : `*~`, `.directory`

## ✅ Fichiers CONSERVÉS dans Git

### Fichiers essentiels du projet web
```
Code/WEB/
├── data_scores_0_10.json         ✅ (scores normalisés)
├── donnees_completes_communes.json ✅ (données chatbot)
├── lisa_data.js                  ✅ (constantes LISA 5%)
├── lisa_data_1pct.js             ✅ (constantes LISA 1%)
├── cah_data.js                   ✅ (constantes CAH)
├── cah_3_clusters.json           ✅ (données CAH 3 clusters)
├── cah_5_clusters.json           ✅ (données CAH 5 clusters)
├── OppChoVec.html                ✅ (page principale)
├── script.js                     ✅ (logique JavaScript)
├── dashboard.css                 ✅ (styles)
├── cah_3_clusters_ecarts.png     ✅ (graphique web)
└── cah_5_clusters_ecarts.png     ✅ (graphique web)
```

### Fichiers de configuration
- `Données/data_indicateurs.json` ✅ (indicateurs bruts)
- `README.md`, `.gitignore`
- Scripts Python (`.py`)
- Fichiers HTML, CSS, JavaScript

## 📦 Taille approximative économisée

Sans le `.gitignore`, le dépôt Git inclurait :
- **~200+ fichiers Excel** (plusieurs centaines de MB)
- **~100+ images PNG** (plusieurs dizaines de MB)
- **GeoJSON** (~5MB chacun)
- **Archives et doublons** (~50MB+)

**Total économisé : > 500 MB** 🎉

## 🔄 Pour récupérer les données ignorées

Si quelqu'un clone le dépôt et a besoin des données complètes :

1. **Générer les données** :
   ```bash
   cd Code/Python
   python generer_donnees_communes_chatbot.py
   python preparer_donnees_cah_web_complet.py
   ```

2. **Télécharger les GeoJSON** (si nécessaire) :
   - Source : [Data.gouv.fr - Contours communes Corse]

3. **Recalculer les analyses** :
   ```bash
   python calculer_moran_queen.py
   python cah_3_dimensions.py
   ```

## 📝 Notes

- Les fichiers essentiels pour faire fonctionner l'application web sont **tous conservés**
- Les données brutes sources doivent être régénérées ou téléchargées
- Les résultats d'analyse (OUTPUT) sont reproductibles via les scripts Python
