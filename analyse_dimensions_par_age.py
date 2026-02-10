"""
Analyse des dimensions de qualité de vie par tranche d'âge
Génère un tableau récapitulatif des préoccupations par ordre d'importance
"""

import pandas as pd
import os
from collections import Counter

# Chemins
CSV_PATH = "donnees_brutes/verbatims_par_commune_avec_portrait/_tous_verbatims_avec_portrait.csv"
OUTPUT_PATH = "comparaisons_rag/dimensions_par_tranche_age.xlsx"

# Mapping des tranches d'âge
def age_to_range(age):
    """Convertit l'âge en catégorie"""
    if pd.isna(age):
        return "Non spécifié"
    try:
        age_int = int(float(str(age).strip().replace("plus de 70 ans", "71")))
    except:
        return "Non spécifié"

    if age_int < 25:
        return "15-24 ans (Jeunes)"
    elif age_int < 35:
        return "25-34 ans (Jeunes adultes)"
    elif age_int < 50:
        return "35-49 ans (Adultes)"
    elif age_int < 65:
        return "50-64 ans (Jeunes seniors)"
    else:
        return "65+ ans (Seniors)"

def main():
    print("="*60)
    print("ANALYSE DES DIMENSIONS PAR TRANCHE D'ÂGE")
    print("="*60)

    # 1. Charger les données
    print(f"\n1. Chargement des données depuis {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f"   {len(df)} verbatims chargés")

    # 2. Ajouter la tranche d'âge
    df['tranche_age'] = df['age'].apply(age_to_range)

    # 3. Analyser les dimensions par tranche d'âge
    print("\n2. Analyse des dimensions par tranche d'âge...")

    tranches = [
        "15-24 ans (Jeunes)",
        "25-34 ans (Jeunes adultes)",
        "35-49 ans (Adultes)",
        "50-64 ans (Jeunes seniors)",
        "65+ ans (Seniors)"
    ]

    # Dictionnaire pour stocker les résultats
    results_by_age = {}

    for tranche in tranches:
        df_tranche = df[df['tranche_age'] == tranche]
        n_total = len(df_tranche)

        if n_total == 0:
            continue

        # Compter les dimensions
        dimension_counts = Counter(df_tranche['dimension'].dropna())

        # Calculer les pourcentages et trier
        dimensions_ranked = []
        for dim, count in dimension_counts.most_common():
            pct = (count / n_total) * 100
            dimensions_ranked.append({
                'dimension': dim,
                'count': count,
                'percentage': pct
            })

        results_by_age[tranche] = {
            'total': n_total,
            'dimensions': dimensions_ranked
        }

        print(f"\n   {tranche}: {n_total} verbatims")
        for i, d in enumerate(dimensions_ranked[:5], 1):
            print(f"      {i}. {d['dimension']}: {d['count']} ({d['percentage']:.1f}%)")

    # 4. Créer le tableau récapitulatif
    print("\n3. Création du tableau récapitulatif...")

    # Feuille 1: Tableau croisé (tranches en colonnes, dimensions en lignes)
    all_dimensions = sorted(set(df['dimension'].dropna()))

    tableau_croise = []
    for dim in all_dimensions:
        row = {'Dimension': dim}
        for tranche in tranches:
            if tranche in results_by_age:
                dim_data = next((d for d in results_by_age[tranche]['dimensions'] if d['dimension'] == dim), None)
                if dim_data:
                    row[tranche] = f"{dim_data['count']} ({dim_data['percentage']:.1f}%)"
                else:
                    row[tranche] = "0 (0%)"
            else:
                row[tranche] = "-"
        tableau_croise.append(row)

    df_croise = pd.DataFrame(tableau_croise)

    # Feuille 2: Top 5 dimensions par tranche d'âge
    top5_data = []
    for tranche in tranches:
        if tranche not in results_by_age:
            continue
        for rank, d in enumerate(results_by_age[tranche]['dimensions'][:5], 1):
            top5_data.append({
                'Tranche d\'âge': tranche,
                'Rang': rank,
                'Dimension': d['dimension'],
                'Nombre': d['count'],
                'Pourcentage': f"{d['percentage']:.1f}%"
            })

    df_top5 = pd.DataFrame(top5_data)

    # Feuille 3: Tableau pivot (rangs par tranche)
    pivot_data = []
    for rank in range(1, 6):  # Top 5
        row = {'Rang': rank}
        for tranche in tranches:
            if tranche in results_by_age and len(results_by_age[tranche]['dimensions']) >= rank:
                d = results_by_age[tranche]['dimensions'][rank-1]
                row[tranche] = f"{d['dimension']} ({d['percentage']:.1f}%)"
            else:
                row[tranche] = "-"
        pivot_data.append(row)

    df_pivot = pd.DataFrame(pivot_data)

    # Feuille 4: Statistiques par genre et tranche d'âge
    stats_genre = []
    for tranche in tranches:
        df_tranche = df[df['tranche_age'] == tranche]
        for genre in ['Femme', 'Homme']:
            df_genre = df_tranche[df_tranche['genre'] == genre]
            n = len(df_genre)
            if n == 0:
                continue
            top_dims = Counter(df_genre['dimension'].dropna()).most_common(3)
            stats_genre.append({
                'Tranche d\'âge': tranche,
                'Genre': genre,
                'Nombre': n,
                'Top 1': f"{top_dims[0][0]} ({top_dims[0][1]})" if len(top_dims) > 0 else "-",
                'Top 2': f"{top_dims[1][0]} ({top_dims[1][1]})" if len(top_dims) > 1 else "-",
                'Top 3': f"{top_dims[2][0]} ({top_dims[2][1]})" if len(top_dims) > 2 else "-",
            })

    df_genre = pd.DataFrame(stats_genre)

    # Feuille 5: Statistiques par profession et tranche d'âge
    stats_profession = []
    for tranche in tranches:
        df_tranche = df[df['tranche_age'] == tranche]
        professions = df_tranche['profession'].dropna().unique()
        for prof in professions:
            df_prof = df_tranche[df_tranche['profession'] == prof]
            n = len(df_prof)
            if n < 3:  # Ignorer si trop peu de données
                continue
            top_dims = Counter(df_prof['dimension'].dropna()).most_common(3)
            stats_profession.append({
                'Tranche d\'âge': tranche,
                'Profession': prof,
                'Nombre': n,
                'Top 1': f"{top_dims[0][0]}" if len(top_dims) > 0 else "-",
                'Top 2': f"{top_dims[1][0]}" if len(top_dims) > 1 else "-",
                'Top 3': f"{top_dims[2][0]}" if len(top_dims) > 2 else "-",
            })

    df_profession = pd.DataFrame(stats_profession)

    # 5. Exporter en Excel
    print(f"\n4. Export vers {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        # Feuille résumé pivot
        df_pivot.to_excel(writer, sheet_name='Top5_par_tranche', index=False)

        # Tableau croisé complet
        df_croise.to_excel(writer, sheet_name='Tableau_croisé', index=False)

        # Top 5 détaillé
        df_top5.to_excel(writer, sheet_name='Top5_détaillé', index=False)

        # Par genre
        df_genre.to_excel(writer, sheet_name='Par_genre', index=False)

        # Par profession
        if len(df_profession) > 0:
            df_profession.to_excel(writer, sheet_name='Par_profession', index=False)

        # Ajuster largeur colonnes
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column)
                adjusted_width = min(max_length + 2, 40)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"\n   OK Fichier exporté: {OUTPUT_PATH}")

    # 6. Afficher le résumé
    print("\n" + "="*60)
    print("RÉSUMÉ: TOP 3 DIMENSIONS PAR TRANCHE D'ÂGE")
    print("="*60)

    for tranche in tranches:
        if tranche not in results_by_age:
            continue
        print(f"\n{tranche} (n={results_by_age[tranche]['total']}):")
        for i, d in enumerate(results_by_age[tranche]['dimensions'][:3], 1):
            print(f"   {i}. {d['dimension']}: {d['percentage']:.1f}%")

    print("\n" + "="*60)
    print("ANALYSE TERMINÉE")
    print("="*60)

if __name__ == "__main__":
    main()
