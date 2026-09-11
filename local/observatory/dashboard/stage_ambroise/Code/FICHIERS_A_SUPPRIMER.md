# Fichiers à supprimer - Analyse du projet

## 📊 Résumé

- **Fichiers Python totaux** : 29
- **Fichiers Excel totaux** : 13 (Code/Python)
- **Fichiers WEB** : 14

## ✅ Fichiers ESSENTIELS (À GARDER)

### Code/WEB (7 fichiers essentiels)
- ✅ **OppChoVec.html** - Page principale de la carte
- ✅ **script.js** - Logique JavaScript
- ✅ **dashboard.css** - Styles utilisés par OppChoVec.html
- ✅ **data_indicateurs.json** - Données utilisées par la carte
- ✅ **data_scores.json** - Scores calculés
- ✅ **Commune_Corse.geojson** - Géométries des communes
- ✅ **seuils_jenks_final.json** - Seuils actuellement utilisés

### Code/Python (Scripts principaux)
- ✅ **oppchovec.py** - Script principal de calcul
- ✅ **calculer_seuils_jenks.py** - Calcul des seuils de Jenks
- ✅ **lisa.py** - Analyse spatiale LISA
- ✅ **services_accessibles_20min.py** - Calcul accessibilité services
- ✅ **verifier_coherence_donnees.py** - Vérification (créé récemment)
- ✅ **creer_data_scores.py** - Génération data_scores.json

### Code/Python (Fichiers de données)
- ✅ **data_indicateurs.json** - Données complètes
- ✅ **data_scores.json** - Scores normalisés
- ✅ **oppchovec_resultats_V.xlsx** - Résultats de référence
- ✅ **donnees_oppchovec_par_dimension.xlsx** - Tableau principal
- ✅ **services_accessibles_20min.csv** - Données d'accessibilité
- ✅ **communes_corse_coordonnees.csv** - Coordonnées GPS

### Données/Corse_Commune (Sources)
- ✅ **Opp1.xlsx, Opp2.xlsx, Opp3.xlsx, Opp4.xlsx** - Données Opportunités
- ✅ **Cho1.xlsx, Cho2.xlsx** - Données Choix
- ✅ **Vec1.xlsx, Vec2.xlsx, Vec3.xlsx, Vec4.xlsx** - Données Vécu
- ✅ **vec2_lb.csv** - Données Vec2 détaillées
- ✅ **mapping_communes.csv** - Mapping codes INSEE

---

## ❌ FICHIERS À SUPPRIMER

### Code/Python - Scripts de test/debug (11 fichiers)
```
❌ test_vec2.py                          # Test ancien
❌ test_oppchovec_complet.py            # Test ancien
❌ test_opp3_opp4_corrected.py          # Test ancien
❌ test_cho2_correction.py              # Test ancien
❌ test_osrm_robuste.py                 # Test ancien
❌ check_oppchovec.py                   # Vérification obsolète
❌ check_opp3_opp4.py                   # Vérification obsolète
❌ verif_excel.py                       # Vérification obsolète
❌ verifier_cho2.py                     # Vérification obsolète
❌ verification_formule_oppchovec.py    # Doublon avec verifier_coherence_donnees.py
❌ analyser_oppchovec_max.py            # Analyse obsolète
❌ analyser_oppchovec_max_v2.py         # Analyse obsolète (v2)
```

### Code/Python - Scripts d'export redondants (5 fichiers)
```
❌ export_indicateurs_disponibles.py    # Export ancien
❌ export_complet_avec_oppchovec.py     # Export obsolète
❌ generer_excel_complet.py             # Génération obsolète
❌ generer_excel_par_dimension.py       # Génération obsolète
❌ creer_excel_feuilles_separees.py     # Génération obsolète
```

### Code/Python - Scripts obsolètes (5 fichiers)
```
❌ services_accessibles_20min_optimise.py   # Ancienne version
❌ services_accessibles_20min_robuste.py    # Ancienne version
❌ services_accessibles_rapide.py           # Ancienne version
❌ temps_acces_services.py                  # Obsolète
❌ afficher_accessibilite.py                # Obsolète
❌ normaliser_oppchovec_sur_10.py           # Tâche déjà faite
```

### Code/Python - Fichiers Excel redondants (9 fichiers)
```
❌ resultats_complets_avec_oppchovec.xlsx       # Doublon (55.8 KB)
❌ resultats_complets_oppchovec_feuilles.xlsx   # Doublon (91.1 KB)
❌ resultats_oppchovec_complet.xlsx             # Doublon (48.8 KB)
❌ df_indicateur.xlsx                           # Doublon (34.7 KB)
❌ dimensions_V.xlsx                            # Doublon (19.7 KB)
❌ Comparaison_V.xlsx                           # Ancien (5.8 KB)
❌ stats_descriptives_normalisées_V.xlsx        # Ancien (5.0 KB)
❌ oppchovec_resultats_normalisées_V.xlsx       # Doublon (32.4 KB)
❌ services_accessibles_20min.xlsx              # Doublon du CSV (32.9 KB)
```
**Total à économiser : ~326 KB**

### Code/Python - Fichiers de données obsolètes (4 fichiers)
```
❌ temps_acces_services.json                # Obsolète
❌ temps_acces_services_corse.csv           # Obsolète
❌ temps_acces_services_corse.xlsx          # Obsolète
❌ services_accessibles_20min.json          # Doublon du CSV
❌ test_communes_10.csv                     # Fichier de test
❌ seuils_jenks.json                        # Ancien (en Python/)
❌ seuils_jenks.js                          # Ancien (en Python/)
❌ indicateurs_corse_vec2_mis_a_jour.xlsx   # Ancien (66.2 KB)
❌ bpe_corse.csv                            # Source brute non utilisée
```

### Code/WEB - Fichiers obsolètes (5 fichiers)
```
❌ seuils_jenks.json                    # Ancien (6.7 KB)
❌ seuils_jenks.js                      # Ancien (0.9 KB)
❌ seuils_jenks_4classes.json           # Version intermédiaire (0.3 KB)
❌ style.css                            # Remplacé par dashboard.css (2.1 KB)
❌ README.md                            # Documentation redondante
❌ README_JENKS.md                      # Documentation redondante
```

### Données/Corse_Commune - Doublons (3 fichiers)
```
❌ oppchovec_resultats.xlsx             # Ancien
❌ oppchovec_resultats_P.xlsx           # Ancienne version (P)
❌ Vec2.xlsx                            # Doublon de vec2_lb.csv
```

---

## 📈 Espace disque à libérer

| Catégorie | Nombre de fichiers | Espace estimé |
|-----------|-------------------|---------------|
| Scripts Python | 26 | ~100 KB |
| Fichiers Excel | 12 | ~450 KB |
| JSON/CSV | 9 | ~100 KB |
| CSS/JS | 4 | ~10 KB |
| **TOTAL** | **51 fichiers** | **~660 KB** |

---

## ⚠️ ATTENTION - Fichiers à GARDER dans Données/

### NE PAS SUPPRIMER :
- ✅ **mapping_communes.csv** - Utilisé par oppchovec.py
- ✅ **vec2_lb.csv** - Utilisé par oppchovec.py
- ✅ **oppchovec_resultats_V.xlsx** - Version finale de référence
- ✅ Tous les fichiers **Opp*.xlsx, Cho*.xlsx, Vec*.xlsx** (sauf Vec2.xlsx)

---

## 🚀 Actions recommandées

### Option 1 : Suppression prudente (recommandé)
Supprimer uniquement les fichiers de test et les anciens scripts (~26 fichiers Python)

### Option 2 : Suppression complète
Supprimer tous les fichiers listés ci-dessus (~51 fichiers)

### Option 3 : Archivage
Déplacer les fichiers dans un dossier `Archive/` au lieu de les supprimer

---

## 📝 Note importante

Avant de supprimer, assurez-vous que :
1. ✅ La carte web fonctionne correctement
2. ✅ Vous avez une sauvegarde du projet
3. ✅ Vous pouvez recalculer OppChoVec avec oppchovec.py
4. ✅ Vous pouvez régénérer les seuils de Jenks

---

*Rapport généré le 2025-01-06*
