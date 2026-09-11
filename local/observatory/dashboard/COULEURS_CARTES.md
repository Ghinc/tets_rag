# 🎨 Guide des couleurs des cartes

Ce document récapitule tous les endroits où les couleurs des cartes sont définies pour que tu puisses facilement les modifier et tester de nouvelles palettes !

---

## 📍 Localisation des définitions de couleurs

### 1. **Cartes OppChoLiv, Opp, Cho, Liv** (Cartes choroplèthes avec Jenks)
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Ligne:** ~92

```javascript
// Palette de couleurs (5 classes) - Bleu clair vers Violet (inversé pour valeurs croissantes)
const colorsJenks = ["#bbdefb", "#64b5f6", "#9c27b0", "#7b1fa2", "#4a148c"];
```

**Description:**
- 5 couleurs pour 5 classes de Jenks
- Dégradé : Bleu clair (valeurs faibles) → Violet foncé (valeurs élevées)
- Utilisé pour toutes les cartes choroplèthes (OppChoLiv, Opp, Cho, Liv)

**Palette actuelle:**
- `#bbdefb` - Bleu très clair (classe 1)
- `#64b5f6` - Bleu clair (classe 2)
- `#9c27b0` - Violet (classe 3)
- `#7b1fa2` - Violet foncé (classe 4)
- `#4a148c` - Violet très foncé (classe 5)

---

### 2. **Cartes LISA** (Local Indicators of Spatial Association)
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Ligne:** ~802

```javascript
// Palette de couleurs LISA
const colorsLISA = {
    'Non significatif': '#f3f3f3',
    'HH (High-High)': '#d73027',
    'LL (Low-Low)': '#4575b4',
    'LH (Low-High)': '#abd9e9',
    'HL (High-Low)': '#fdae61'
};
```

**Description:**
- 5 catégories de clusters spatiaux
- Palette standard pour l'analyse LISA

**Palette actuelle:**
- `#f3f3f3` - Gris très clair (Non significatif)
- `#d73027` - Rouge (HH - Hotspots)
- `#4575b4` - Bleu foncé (LL - Coldspots)
- `#abd9e9` - Bleu clair (LH - Outlier négatif)
- `#fdae61` - Orange (HL - Outlier positif)

**Aussi défini dans la légende (ligne ~884):**
```javascript
const categories = [
    { color: '#d73027', labelFr: 'HH (High-High)', labelEn: 'HH (High-High)', ... },
    { color: '#4575b4', labelFr: 'LL (Low-Low)', labelEn: 'LL (Low-Low)', ... },
    { color: '#fdae61', labelFr: 'HL (High-Low)', labelEn: 'HL (High-Low)', ... },
    { color: '#abd9e9', labelFr: 'LH (Low-High)', labelEn: 'LH (Low-High)', ... },
    { color: '#d9d9d9', labelFr: 'Non significatif', labelEn: 'Not significant', ... }
];
```

---

### 3. **Cartes CAH** (Classification Hiérarchique Ascendante)
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Ligne:** ~1017

```javascript
// Palette de couleurs CAH (jusqu'à 5 clusters)
const colorsCAH = {
    1: '#917648',  // Marron - Cluster 1
    2: '#eee0ba',  // Beige - Cluster 2
    3: '#61e75c',  // Vert - Cluster 3
    4: '#de7eed',  // Violet - Cluster 4
    5: '#f4b474'   // Orange - Cluster 5
};
```

**Description:**
- Palette qualitative pour distinguer les clusters
- Palette personnalisée (marron-beige-vert-violet-orange)
- Utilisée pour CAH 3 clusters ET CAH 5 clusters

**Palette actuelle:**
- `#917648` - Marron (Cluster 1)
- `#eee0ba` - Beige (Cluster 2)
- `#61e75c` - Vert (Cluster 3)
- `#de7eed` - Violet (Cluster 4)
- `#f4b474` - Orange (Cluster 5)

**Aussi défini dans la légende (ligne ~1107):**
```javascript
const colorsCAH = {
    1: '#917648',  // Marron
    2: '#eee0ba',  // Beige
    3: '#61e75c',  // Vert
    4: '#de7eed',  // Violet
    5: '#f4b474'   // Orange
};
```

---

### 4. **Routes principales**
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Ligne:** ~372

```javascript
const creerCoucheRoute = (geojsonData) => {
    return L.geoJSON(geojsonData, {
        pane: paneName,
        style: {
            color: '#ff0000',  // Rouge
            weight: 2,
            opacity: 0.7
        }
    });
};
```

**Description:**
- Couleur unique pour toutes les routes
- Appliquée aux routes nationales, départementales, communales

**Couleur actuelle:**
- `#ff0000` - Rouge vif

---

### 5. **Limites des communes**
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Lignes:** ~470, 603, 738, 893, 1036

```javascript
// Dans les fonctions styleCommune et style des GeoJSON
{
    color: "#000000",  // Noir (contour normal)
    weight: 1,
    fillOpacity: 0.7
}

// Quand commune survolée
{
    weight: 3,
    color: '#333',  // Gris foncé
    fillOpacity: 0.9
}

// Quand commune sélectionnée
{
    color: "red",  // Rouge
    weight: 3,
    fillOpacity: 0.7
}
```

---

### 6. **Villes principales**
**Fichier:** `stage_ambroise/Code/WEB/script.js`
**Lignes:** ~125-159

```javascript
// Lignes pointillées reliant le point au label
L.polyline([posVille, posLabel], {
    color: '#000000',  // Noir
    weight: 1,
    opacity: 0.6,
    dashArray: '3, 3',
    pane: lignesPaneName
});

// Marqueur de la ville (point)
L.circleMarker(posVille, {
    radius: 5,
    fillColor: "#000000",  // Noir
    color: "#ffffff",      // Contour blanc
    weight: 2,
    opacity: 1,
    fillOpacity: 1
});

// Label du nom de ville
background-color: rgba(255, 255, 255, 0.85);  // Fond blanc semi-transparent
color: #000;  // Texte noir
border: 1px solid #000;  // Bordure noire
```

---

## 🎨 Palettes de couleurs alternatives (suggestions)

### Pour les cartes choroplèthes (OppChoLiv, Opp, Cho, Liv)

**Option 1 - Vert-Jaune (ColorBrewer YlGn):**
```javascript
const colorsJenks = ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"];
```

**Option 2 - Orange-Rouge (ColorBrewer YlOrRd):**
```javascript
const colorsJenks = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"];
```

**Option 3 - Bleu-Vert (ColorBrewer BuGn):**
```javascript
const colorsJenks = ["#edf8fb", "#b2e2e2", "#66c2a4", "#2ca25f", "#006d2c"];
```

**Option 4 - Viridis (palette scientifique):**
```javascript
const colorsJenks = ["#440154", "#31688e", "#35b779", "#fde724", "#fee825"];
```

### Pour les clusters CAH

**Option 1 - Palette Pastel:**
```javascript
const colorsCAH = {
    1: '#fbb4ae',  // Rose pastel
    2: '#b3cde3',  // Bleu pastel
    3: '#ccebc5',  // Vert pastel
    4: '#decbe4',  // Violet pastel
    5: '#fed9a6'   // Orange pastel
};
```

**Option 2 - Palette Vive (ColorBrewer Dark2):**
```javascript
const colorsCAH = {
    1: '#1b9e77',  // Turquoise
    2: '#d95f02',  // Orange
    3: '#7570b3',  // Violet
    4: '#e7298a',  // Rose
    5: '#66a61e'   // Vert
};
```

---

## 💡 Conseils pour modifier les couleurs

1. **Test rapide:** Modifie directement dans `script.js` et recharge la page (Ctrl+F5)

2. **Vérifier le contraste:** Assure-toi que les couleurs sont bien visibles sur fond blanc

3. **Accessibilité:** Vérifie que les couleurs sont distinguables pour les daltoniens
   - Outils: https://colorbrewer2.org (option "colorblind safe")
   - Ou: https://davidmathlogic.com/colorblind/

4. **Cohérence:** Garde la même philosophie de couleur entre LISA 5% et LISA 1%

5. **Export PNG:** Les nouvelles couleurs seront automatiquement incluses dans les exports PNG

---

## 📝 Note importante

Si tu changes les couleurs CAH ou LISA, pense à les modifier à **deux endroits** dans `script.js`:
1. Dans la fonction principale (`afficherCarteLISA` ou `afficherCarteCAH`)
2. Dans la fonction de légende (`ajouterLegendeLISA` ou `ajouterLegendeCAH`)

Bonne expérimentation ! 🎨
