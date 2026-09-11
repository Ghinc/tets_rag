# Vérification de la cohérence des données OppChoVec

## Résumé

**✅ Les données sont IDENTIQUES entre la carte web et le tableau `donnees_oppchovec_par_dimension.xlsx`**

Une vérification complète a été effectuée sur les **360 communes** et **aucune différence** n'a été détectée.

## Fichiers comparés

1. **donnees_oppchovec_par_dimension.xlsx** (tableau de référence)
2. **data_indicateurs.json** (fichier utilisé par la carte web)
3. **oppchovec_resultats_V.xlsx** (résultats complets)

## Formules utilisées

### Python (oppchovec.py)
```python
ALPHA = 2.5
BETA = 1.5
Pondérations Opp: [0.25, 0.25, 0.25, 0.25]
Pondérations Cho: [0.50, 0.50]
Pondérations Vec: [0.25, 0.25, 0.25, 0.25]

# Formule OppChoVec
somme_ponderee = sum(pk * (dik ** BETA))
oppchovec = (1/3) * (somme_ponderee ** (ALPHA / BETA))
```

### JavaScript (script.js)
```javascript
const alpha = 2.5;
const beta = 1.5;
// Pondérations identiques

// Formule OppChoVec (identique à Python)
const dik = scores.map(score => Math.pow(score, beta));
const sommePonderee = sum(pk * dik[i]);
const indice = (1/3) * Math.pow(sommePonderee, alpha / beta);
```

## Exemple de vérification : Ajaccio

| Indicateur | Tableau Excel | Carte Web | Différence |
|------------|---------------|-----------|------------|
| **OppChoVec** | 1.368496 | 1.368496 | **0.000000** |
| **Score_Opp** | 9.621243 | 9.621243 | **0.000000** |
| **Score_Cho** | 0.547654 | 0.547654 | **0.000000** |
| **Score_Vec** | 4.581839 | 4.581839 | **0.000000** |
| Opp1 (Éducation) | 5.76 | 5.76 | 0.00 |
| Opp2 (Theil) | 0.278668 | 0.278668 | 0.000000 |
| Opp3 (Transport) | 186.1 | 186.1 | 0.0 |
| Opp4 (Connectivité) | 93.5 | 93.5 | 0.0 |

## Si vous observez des différences

### Cas possibles :

1. **Versions de fichiers différentes**
   - Vérifiez que vous utilisez les fichiers les plus récents
   - Date de modification des fichiers

2. **Confusion entre valeurs normalisées et non-normalisées**
   - Les **indicateurs bruts** (Opp1, Opp2, etc.) ne sont PAS normalisés
   - Les **scores de dimensions** (Score_Opp, Score_Cho, Score_Vec) sont normalisés sur **0-10**
   - L'**indice OppChoVec** final est calculé à partir des scores normalisés

3. **Confusion entre anciennes et nouvelles versions**
   - Ancien fichier : valeurs normalisées sur 0-1
   - Nouveau fichier : valeurs normalisées sur 0-10
   - Vérifiez quelle échelle vous comparez

4. **Arrondis d'affichage**
   - Les valeurs affichées peuvent être arrondies (2 décimales)
   - Les valeurs stockées ont 6+ décimales de précision
   - Cela peut donner l'impression de différences minimes

## Comment vérifier vous-même

### Méthode 1 : Script Python
```bash
cd "Code/Python"
python verifier_coherence_donnees.py
```

### Méthode 2 : Vérification manuelle
1. Ouvrir `donnees_oppchovec_par_dimension.xlsx`
2. Ouvrir la carte web dans le navigateur
3. Sélectionner une commune (ex: Ajaccio)
4. Comparer les valeurs affichées
5. Vérifier que les décimales correspondent

### Méthode 3 : Inspecter le JSON
```javascript
// Ouvrir la console du navigateur (F12)
// Taper dans la console :
console.log(indiceFinale["Ajaccio"]);
console.log(scoresParCommune["Ajaccio"]);
```

## Échelles utilisées

### Indicateurs bruts (NON normalisés)
- **Opp1** : 1 à 7 (niveau d'éducation)
- **Opp2** : 0 à 1 (indice de Theil)
- **Opp3** : 0 à 200 (accès transport, en %)
- **Opp4** : 0 à 100 (connectivité, en %)
- **Cho1** : 0 à 1 (exp(-quartiers))
- **Cho2** : 0 à 100 (participation électorale, en %)
- **Vec1** : 15000 à 30000 (revenu médian, en €)
- **Vec2** : 0 à 1 (qualité logement)
- **Vec3** : 1 à 5 (stabilité emploi)
- **Vec4** : 0 à 5000 (nb services accessibles)

### Scores normalisés (NORMALISÉS sur 0-10)
- **Score_Opp** : 0 à 10
- **Score_Cho** : 0 à 10
- **Score_Vec** : 0 à 10
- **OppChoVec** : 0 à 10 (indice final)

## Contact

Si vous constatez toujours des différences après ces vérifications, merci de préciser :
1. **Quelle commune** présente une différence
2. **Quel indicateur** diffère
3. **Quelles valeurs** vous voyez (tableau vs carte)
4. **Quelle version** des fichiers vous utilisez

---
*Document généré automatiquement le 2025-01-06*
*Script de vérification : `verifier_coherence_donnees.py`*
