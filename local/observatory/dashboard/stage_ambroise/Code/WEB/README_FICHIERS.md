# Fichiers pour l'application web (normalisation 0-10)

## 📁 Fichier principal à charger

### **data_scores_0_10.json** ✅
**C'est le fichier que votre application doit charger !**

Contient pour chaque commune :
- `OppChoVec_0_10` : Score global (0-10)
- `Score_Opp_0_10` : Score Opportunités (0-10)
- `Score_Cho_0_10` : Score Choix (0-10)
- `Score_Vec_0_10` : Score Vécu (0-10)
- Tous les sous-indicateurs : Opp1, Opp2, Opp3, Opp4, Cho1, Cho2, Vec1, Vec2, Vec3, Vec4

---

## 🎨 Fichier des seuils pour les couleurs

### **seuils_jenks.json** ✅
Contient les seuils Jenks (5 classes) pour TOUTES les variables :

#### Clés disponibles :
- `OppChoVec_0_10` ou `OppChoVec`
- `Score_Opp_0_10` ou `Score_Opp`
- `Score_Cho_0_10` ou `Score_Cho`
- `Score_Vec_0_10` ou `Score_Vec`

#### Seuils (0-10) :

**OppChoVec :**
```
Classe 1 (Très faible) : 0.00 - 2.41
Classe 2 (Faible)      : 2.41 - 3.93
Classe 3 (Moyen)       : 3.93 - 5.11
Classe 4 (Élevé)       : 5.11 - 7.56
Classe 5 (Très élevé)  : 7.56 - 10.00
```

**Score_Opp :**
```
Classe 1 : 0.00 - 2.47
Classe 2 : 2.47 - 3.77
Classe 3 : 3.77 - 5.04
Classe 4 : 5.04 - 6.57
Classe 5 : 6.57 - 10.00
```

**Score_Cho :**
```
Classe 1 : 0.00 - 3.33
Classe 2 : 3.33 - 6.26
Classe 3 : 6.26 - 8.28
Classe 4 : 8.28 - 9.31
Classe 5 : 9.31 - 10.00
```

**Score_Vec :**
```
Classe 1 : 0.00 - 1.83
Classe 2 : 1.83 - 3.08
Classe 3 : 3.08 - 4.18
Classe 4 : 4.18 - 6.70
Classe 5 : 6.70 - 10.00
```

---

## 🗺️ Fichiers LISA (optionnels)

### **lisa_oppchovec_0_10_1pct.json** (seuil 1%)
26 communes significatives (7.2%)

### **lisa_oppchovec_0_10_5pct.json** (seuil 5%)
56 communes significatives (15.6%)

---

## 💡 Comment utiliser dans votre application

### 1. Charger les données
```javascript
const dataScores = await fetch('data_scores_0_10.json').then(r => r.json());
const seuilsJenks = await fetch('seuils_jenks.json').then(r => r.json());
```

### 2. Obtenir la couleur d'une commune
```javascript
const colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850'];

function getColor(score, dimension = 'OppChoVec_0_10') {
  const seuils = seuilsJenks[dimension];

  for (let i = 0; i < seuils.length - 1; i++) {
    if (score >= seuils[i] && score < seuils[i + 1]) {
      return colors[i];
    }
  }
  return colors[colors.length - 1];
}

// Utilisation
const scoreCommune = dataScores['Ajaccio'].OppChoVec_0_10;
const couleur = getColor(scoreCommune, 'OppChoVec_0_10');
```

### 3. Appliquer aux communes
```javascript
communes.forEach(commune => {
  const nomCommune = commune.properties.nom;
  const score = dataScores[nomCommune]?.OppChoVec_0_10;

  if (score !== undefined) {
    commune.setStyle({
      fillColor: getColor(score, 'OppChoVec_0_10'),
      fillOpacity: 0.7
    });
  }
});
```

---

## ✅ Vérification

Pour vérifier que tout fonctionne :

1. **Console navigateur (F12)** : Vérifier qu'il n'y a pas d'erreurs
2. **Test rapide** : `console.log(dataScores['Afa'].OppChoVec_0_10)` devrait afficher `10`
3. **Vérifier les seuils** : `console.log(seuilsJenks.OppChoVec_0_10)` devrait afficher 6 valeurs

---

## 🎯 Fichier à charger selon votre besoin

| Besoin | Fichier | Clé à utiliser |
|--------|---------|----------------|
| Carte OppChoVec | `data_scores_0_10.json` | `OppChoVec_0_10` |
| Carte Opportunités | `data_scores_0_10.json` | `Score_Opp_0_10` |
| Carte Choix | `data_scores_0_10.json` | `Score_Cho_0_10` |
| Carte Vécu | `data_scores_0_10.json` | `Score_Vec_0_10` |
| Seuils Jenks | `seuils_jenks.json` | `OppChoVec_0_10` ou `Score_Opp_0_10` etc. |
| LISA 1% | `lisa_oppchovec_0_10_1pct.json` | - |
| LISA 5% | `lisa_oppchovec_0_10_5pct.json` | - |

---

## ⚠️ Important

- Toujours utiliser les clés avec `_0_10` : `OppChoVec_0_10`, `Score_Opp_0_10`, etc.
- Les seuils Jenks contiennent 6 valeurs (pour 5 classes)
- Toutes les valeurs sont entre 0 et 10
