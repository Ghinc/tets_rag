# Guide d'intégration des données 0-10 dans l'application web

## 📁 Fichiers générés

### Dans `../WEB/`
1. **data_scores_0_10.json** - Données des communes avec tous les scores
2. **seuils_jenks_0_10.json** - Seuils Jenks pour les cartes choroplèthes
3. **lisa_oppchovec_0_10_1pct.json** - Clusters LISA au seuil 1%
4. **lisa_oppchovec_0_10_5pct.json** - Clusters LISA au seuil 5%
5. **config_0_10.json** - Configuration de l'application

### Dans `../Python/`
6. **tableau_indicateurs_0_10.xlsx** - Tableau Excel avec 4 feuilles

## 🎨 Seuils Jenks (5 classes)

Les seuils pour la carte choroplèthe OppChoVec (0-10) :
```
Classe 1 (Très faible) : 0.00 - 2.41
Classe 2 (Faible)      : 2.41 - 3.93
Classe 3 (Moyen)       : 3.93 - 5.11
Classe 4 (Élevé)       : 5.11 - 7.56
Classe 5 (Très élevé)  : 7.56 - 10.00
```

## 🗺️ Structure des données

### data_scores_0_10.json
```json
{
  "Nom_Commune": {
    "OppChoVec_0_10": float,      // Score OppChoVec normalisé 0-10
    "Score_Opp_0_10": float,      // Score Opp normalisé 0-10
    "Score_Cho_0_10": float,      // Score Cho normalisé 0-10
    "Score_Vec_0_10": float,      // Score Vec normalisé 0-10
    "Opp1": float,                // Éducation
    "Opp2": float,                // Diversité sociale (Theil)
    "Opp3": float,                // Transports
    "Opp4": float,                // Connectivité numérique
    "Cho1": float,                // Quartiers prioritaires
    "Cho2": float,                // Droit de vote
    "Vec1": float,                // Revenu médian
    "Vec2": float,                // Qualité logement
    "Vec3": float,                // Stabilité emploi
    "Vec4": float                 // Accès services
  }
}
```

### seuils_jenks_0_10.json
```json
{
  "OppChoVec_0_10": [0.0, 2.41, 3.93, 5.11, 7.56, 10.0]
}
```

### lisa_oppchovec_0_10_1pct.json et 5pct.json
```json
{
  "Nom_Commune": {
    "cluster": "HH (High-High)" | "LL (Low-Low)" | "HL (High-Low)" | "LH (Low-High)" | "Non significatif",
    "p_value": float,
    "I": float,
    "significatif": boolean,
    "quadrant": int (1-4)
  }
}
```

## 🎯 Utilisation dans l'application web

### 1. Chargement des données

```javascript
// Charger les données principales
const dataScores = await fetch('data_scores_0_10.json').then(r => r.json());

// Charger les seuils Jenks
const seuilsJenks = await fetch('seuils_jenks_0_10.json').then(r => r.json());

// Charger LISA
const lisa1pct = await fetch('lisa_oppchovec_0_10_1pct.json').then(r => r.json());
const lisa5pct = await fetch('lisa_oppchovec_0_10_5pct.json').then(r => r.json());
```

### 2. Colorisation de la carte

```javascript
// Fonction pour obtenir la couleur selon le score
function getColorFromScore(score, seuils) {
  const colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850'];

  for (let i = 0; i < seuils.length - 1; i++) {
    if (score >= seuils[i] && score < seuils[i + 1]) {
      return colors[i];
    }
  }
  return colors[colors.length - 1];
}

// Appliquer la couleur à chaque commune
communes.forEach(commune => {
  const score = dataScores[commune.nom]?.OppChoVec_0_10;
  if (score !== undefined) {
    commune.style.fill = getColorFromScore(score, seuilsJenks.OppChoVec_0_10);
  }
});
```

### 3. Affichage des clusters LISA

```javascript
// Couleurs pour les clusters LISA
const lisaColors = {
  'HH (High-High)': '#d7191c',
  'LL (Low-Low)': '#2b83ba',
  'HL (High-Low)': '#fdae61',
  'LH (Low-High)': '#abd9e9',
  'Non significatif': '#ffffbf'
};

// Appliquer les couleurs LISA
communes.forEach(commune => {
  const lisa = lisa1pct[commune.nom];
  if (lisa && lisa.significatif) {
    commune.style.fill = lisaColors[lisa.cluster];
  }
});
```

## 📊 Tableau Excel (tableau_indicateurs_0_10.xlsx)

### Feuille 1 : OppChoVec
- Zone
- OppChoVec_0_10 (triée décroissante)
- Score_Opp_0_10
- Score_Cho_0_10
- Score_Vec_0_10

### Feuille 2 : Opp (Opportunités)
- Zone
- Score_Opp_0_10 (triée décroissante)
- Opp1 (Éducation)
- Opp2 (Diversité sociale)
- Opp3 (Transports)
- Opp4 (Connectivité)

### Feuille 3 : Cho (Choix)
- Zone
- Score_Cho_0_10 (triée décroissante)
- Cho1 (Quartiers prioritaires)
- Cho2 (Droit de vote)

### Feuille 4 : Vec (Vécu)
- Zone
- Score_Vec_0_10 (triée décroissante)
- Vec1 (Revenu)
- Vec2 (Logement)
- Vec3 (Emploi)
- Vec4 (Services)

## 🔍 Vérification

Pour vérifier que tout fonctionne :

1. **Vérifier que les seuils Jenks ont 6 valeurs** (pour 5 classes)
2. **Vérifier que toutes les communes ont un score OppChoVec_0_10**
3. **Vérifier que les noms de communes correspondent** entre les fichiers JSON et les géométries

## 🐛 Résolution du problème de couleurs

Si les couleurs ne s'affichent pas :

1. **Vérifier la correspondance des noms** :
   - Les noms dans `data_scores_0_10.json` doivent correspondre exactement aux noms dans les géométries
   - Attention aux accents, majuscules, tirets

2. **Vérifier la structure des données** :
   - Chaque commune doit avoir une valeur `OppChoVec_0_10`
   - Les valeurs doivent être comprises entre 0 et 10

3. **Vérifier les seuils Jenks** :
   - Doit contenir exactement 6 valeurs : [min, seuil1, seuil2, seuil3, seuil4, max]
   - Le dernier seuil doit être 10.0

4. **Console JavaScript** :
   - Ouvrir la console du navigateur (F12)
   - Vérifier s'il y a des erreurs
   - Vérifier que les données sont bien chargées

## 📝 Notes importantes

- **Normalisation** : Tous les scores sont entre 0 et 10
- **LISA** : Uniquement calculé sur OppChoVec (pas sur Opp, Cho, Vec)
- **Seuils** : Calculés avec l'algorithme de Jenks (Natural Breaks)
- **Communes** : 360 communes de Corse
