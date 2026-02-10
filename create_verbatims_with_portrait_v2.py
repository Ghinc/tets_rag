"""
Script pour créer des fichiers de verbatims par commune avec les données portrait
Version 2: Matching intelligent des dimensions par analyse du contenu
"""
import pandas as pd
import os
import re
from collections import defaultdict

# Dictionnaire de mots-clés pour chaque dimension
DIMENSION_KEYWORDS = {
    'Santé': ['santé', 'sante', 'médecin', 'medecin', 'hôpital', 'hopital', 'hospitalier', 'soins', 'soigner', 'maladie', 'docteur', 'médical', 'medical', 'mourir', 'mort', 'vivre', 'espérance'],
    'Environnement': ['environnement', 'nature', 'air', 'paysage', 'vert', 'pollution', 'écologie', 'ecologie', 'climat', 'durable', 'ressource', 'animaux', 'calme'],
    'Culture': ['culture', 'musée', 'musee', 'théâtre', 'theatre', 'cinéma', 'cinema', 'concert', 'art', 'livre', 'lecture', 'passion', 'loisir', 'divertissement', 'ouverture', 'enrichissement'],
    'Logement': ['logement', 'logé', 'loger', 'maison', 'appartement', 'habitation', 'loyer', 'habitat', 'toit', 'cocon', 'chez soi', 'domicile', 'lumineux', 'place', 'confort', 'dormir'],
    'Services de proximité': ['service', 'commerce', 'magasin', 'courses', 'proximité', 'proximite', 'boutique', 'supermarché', 'epicerie', 'commodité', 'convivialité', 'produit'],
    'Réseau': ['réseau', 'reseau', 'internet', 'téléphone', 'telephone', 'connexion', 'fibre', '4g', '5g', 'wifi', 'couverture', 'papiers', 'administratif', 'cours'],
    'Sécurité': ['sécurité', 'securite', 'sûreté', 'surete', 'tranquillité', 'tranquillite', 'paix', 'calme', 'criminalité', 'violence', 'vol', 'agression', 'serein', 'protéger'],
    'Ratio vie pro/ vie perso': ['équilibre', 'equilibre', 'vie pro', 'vie perso', 'temps libre', 'temps perso', 'personnel', 'balance', 'santé mentale', 'repos', 'coupure', 'ratio'],
    'Education': ['éducation', 'education', 'école', 'ecole', 'fac', 'université', 'universite', 'étude', 'etude', 'apprendre', 'formation', 'compétence', 'diplôme', 'connaissance', 'studià', 'libertà'],
    'Revenus': ['revenu', 'argent', 'salaire', 'euro', 'prix', 'coût', 'cout', 'cher', 'économie', 'economie', 'financ', 'payer', 'capitaliste', 'riche', 'pauvre', 'précaire', 'besoin'],
    'Emploi': ['emploi', 'job', 'métier', 'metier', 'poste', 'carrière', 'embauche', 'chômage', 'recrutement'],
    'Transports': ['transport', 'bus', 'train', 'voiture', 'route', 'trajet', 'déplacement', 'mobilité', 'mobilite', 'permis', 'conduire', 'véhicule', 'locomotion'],
    'Communauté et relations': ['communauté', 'communaute', 'relation', 'famille', 'ami', 'entourage', 'social', 'lien', 'solidarité', 'ensemble', 'voisin', 'partage', 'soutien', 'présent'],
    'Tourisme (ressenti localement)': ['tourisme', 'touriste', 'visiteur', 'attraction', 'vacances', 'île', 'ile'],
    'Démographie': ['démographie', 'demographie', 'population', 'habitant', 'densité', 'vieillissement'],
}

def normalize_text(text):
    """Normalise le texte pour la comparaison"""
    text = text.lower()
    text = re.sub(r'[àâä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[îï]', 'i', text)
    text = re.sub(r'[ôö]', 'o', text)
    text = re.sub(r'[ùûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    return text

def match_dimension(verbatim, available_dimensions, position_index=0):
    """
    Trouve la dimension la plus probable pour un verbatim donné.
    Si incertain, utilise l'ordre positionnel comme fallback.
    Retourne (dimension_matchée, score_confiance, est_certain)
    """
    verbatim_norm = normalize_text(verbatim)

    scores = {}
    for dim in available_dimensions:
        # Chercher la dimension dans notre dictionnaire de mots-clés
        dim_key = None
        for key in DIMENSION_KEYWORDS:
            if normalize_text(key) == normalize_text(dim) or normalize_text(dim) in normalize_text(key):
                dim_key = key
                break

        if dim_key is None:
            # Dimension inconnue, utiliser le nom comme mot-clé
            keywords = [normalize_text(dim)]
        else:
            keywords = [normalize_text(k) for k in DIMENSION_KEYWORDS[dim_key]]

        # Compter les occurrences de mots-clés
        score = 0
        for kw in keywords:
            if kw in verbatim_norm:
                # Bonus si le mot-clé est au début (souvent le sujet principal)
                if verbatim_norm.startswith(kw) or verbatim_norm[:30].find(kw) != -1:
                    score += 3
                else:
                    score += 1

        scores[dim] = score

    # Trouver la meilleure correspondance
    if not scores:
        return available_dimensions[position_index] if position_index < len(available_dimensions) else 'Non spécifié', 0, False

    best_dim = max(scores, key=scores.get)
    best_score = scores[best_dim]

    # Calculer la confiance
    total_score = sum(scores.values())

    # Si aucun mot-clé trouvé, utiliser l'ordre positionnel comme fallback
    if total_score == 0:
        fallback_dim = available_dimensions[position_index] if position_index < len(available_dimensions) else 'Non spécifié'
        return fallback_dim, 0, False

    confidence = best_score / max(total_score, 1)
    is_certain = best_score >= 1 and confidence >= 0.5

    return best_dim, best_score, is_certain

# Charger le fichier CSV
df = pd.read_csv('donnees_brutes/sortie_questionnaire_traited.csv', encoding='utf-8')
cols = df.columns.tolist()

col_genre = cols[2]
col_age = cols[3]
col_profession = cols[4]
col_commune_as = cols[5]
col_commune_tz = cols[6]
col_dimensions = cols[7]
verbatim_cols = [cols[8], cols[9], cols[10]]

# Créer le dossier de sortie
output_dir = 'donnees_brutes/verbatims_par_commune_avec_portrait'
os.makedirs(output_dir, exist_ok=True)

# Statistiques
stats = {'certain': 0, 'incertain': 0, 'total': 0}
uncertain_cases = []

# Dictionnaire pour stocker les verbatims par commune
verbatims_by_commune = defaultdict(list)

print("Traitement des verbatims avec matching intelligent...")

for idx, row in df.iterrows():
    commune = row[col_commune_as] if pd.notna(row[col_commune_as]) and str(row[col_commune_as]).strip() else row[col_commune_tz]

    if pd.isna(commune) or str(commune).strip() == '':
        continue

    commune = str(commune).strip()

    genre = row[col_genre] if pd.notna(row[col_genre]) else 'Non spécifié'
    age = row[col_age] if pd.notna(row[col_age]) else 'Non spécifié'
    profession = row[col_profession] if pd.notna(row[col_profession]) else 'Non spécifié'

    # Extraire les dimensions disponibles
    dimensions_raw = row[col_dimensions] if pd.notna(row[col_dimensions]) else ''
    dimensions_list = [d.strip() for d in str(dimensions_raw).split(';') if d.strip()]

    # Extraire et matcher chaque verbatim
    for i, col in enumerate(verbatim_cols, 1):
        verbatim = row[col] if pd.notna(row[col]) else ''
        verbatim = str(verbatim).strip()

        if verbatim and verbatim.lower() not in ['nan', '', 'none']:
            # Matcher la dimension par contenu (avec fallback positionnel)
            if dimensions_list:
                matched_dim, score, is_certain = match_dimension(verbatim, dimensions_list, position_index=i-1)
            else:
                matched_dim = 'Non spécifié'
                is_certain = False

            stats['total'] += 1
            if is_certain:
                stats['certain'] += 1
            else:
                stats['incertain'] += 1
                if len(uncertain_cases) < 20:  # Garder les 20 premiers cas incertains
                    uncertain_cases.append({
                        'commune': commune,
                        'verbatim': verbatim[:80],
                        'dimensions_dispo': dimensions_list,
                        'matched': matched_dim
                    })

            verbatims_by_commune[commune].append({
                'genre': genre,
                'age': age,
                'profession': profession,
                'dimension': matched_dim,
                'verbatim': verbatim,
                'choix_numero': i,
                'matching_certain': is_certain
            })

print(f"\n=== STATISTIQUES DE MATCHING ===")
print(f"Total verbatims: {stats['total']}")
print(f"Matchings certains: {stats['certain']} ({100*stats['certain']/stats['total']:.1f}%)")
print(f"Matchings incertains: {stats['incertain']} ({100*stats['incertain']/stats['total']:.1f}%)")

if uncertain_cases:
    print(f"\n=== EXEMPLES DE CAS INCERTAINS (premiers {len(uncertain_cases)}) ===")
    for case in uncertain_cases[:10]:
        print(f"\nCommune: {case['commune']}")
        print(f"  Verbatim: {case['verbatim']}...")
        print(f"  Dimensions dispo: {case['dimensions_dispo']}")
        print(f"  Matché à: {case['matched']}")

# Créer les fichiers de sortie
print(f"\nCréation de fichiers pour {len(verbatims_by_commune)} communes...")

for commune, verbatims in verbatims_by_commune.items():
    safe_commune = commune.replace('/', '-').replace('\\', '-').replace(':', '-')
    filename = os.path.join(output_dir, f'{safe_commune}.txt')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"VERBATIMS - COMMUNE DE {commune.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Nombre de verbatims: {len(verbatims)}\n\n")

        for i, v in enumerate(verbatims, 1):
            f.write(f"--- Verbatim {i} ---\n")
            f.write(f"Genre: {v['genre']}\n")
            f.write(f"Âge: {v['age']}\n")
            f.write(f"Profession: {v['profession']}\n")
            certainty = "" if v['matching_certain'] else " [INCERTAIN]"
            f.write(f"Dimension qualité de vie: {v['dimension']}{certainty}\n")
            f.write(f"Verbatim: {v['verbatim']}\n\n")

# Créer le CSV global
all_data = []
for commune, verbatims in verbatims_by_commune.items():
    for v in verbatims:
        all_data.append({
            'commune': commune,
            'genre': v['genre'],
            'age': v['age'],
            'profession': v['profession'],
            'dimension': v['dimension'],
            'matching_certain': v['matching_certain'],
            'verbatim': v['verbatim'],
            'choix_numero': v['choix_numero']
        })

df_output = pd.DataFrame(all_data)
df_output.to_csv(os.path.join(output_dir, '_tous_verbatims_avec_portrait.csv'), index=False, encoding='utf-8')
print(f"\nFichier CSV global créé avec colonne 'matching_certain'")
print(f"Total: {len(all_data)} verbatims")
