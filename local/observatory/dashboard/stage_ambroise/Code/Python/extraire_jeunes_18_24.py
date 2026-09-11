"""
Script pour extraire le nombre de personnes de 18 à 24 ans
pour les communes de Corse à partir du fichier INSEE TD_POP1B_2022
"""

import pandas as pd
import json

def extraire_jeunes_corse(fichier_input='../../Données/TD_POP1B_2022.csv',
                          fichier_csv_output='jeunes_18_24_corse.csv',
                          fichier_json_output='jeunes_18_24_corse.json'):
    """
    Extrait le nombre de personnes de 18 à 24 ans pour chaque commune de Corse
    """
    print("="*80)
    print("EXTRACTION DES 18-24 ANS EN CORSE")
    print("="*80)

    # 1. Charger le fichier
    print(f"\nChargement du fichier {fichier_input}...")
    df = pd.read_csv(fichier_input, sep=';', dtype={'CODGEO': str, 'AGED100': str})
    print(f"  {len(df):,} lignes chargées")

    # 2. Filtrer les communes corses (code commence par 2A ou 2B)
    print("\nFiltrage des communes corses...")
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()
    print(f"  {len(df_corse):,} lignes pour la Corse")
    print(f"  {df_corse['CODGEO'].nunique()} communes distinctes")

    # 3. Filtrer les âges 18 à 24 ans (AGED100 de '018' à '024')
    print("\nFiltrage des 18-24 ans...")
    ages_18_24 = [f'{i:03d}' for i in range(18, 25)]  # ['018', '019', ..., '024']
    df_jeunes = df_corse[df_corse['AGED100'].isin(ages_18_24)].copy()
    print(f"  {len(df_jeunes):,} lignes pour les 18-24 ans")

    # 4. Grouper par commune et sommer (tous sexes confondus)
    print("\nAgrégation par commune...")
    resultat = df_jeunes.groupby(['CODGEO', 'LIBGEO'])['NB'].sum().reset_index()
    resultat.columns = ['Code_INSEE', 'Commune', 'Nb_18_24_ans']
    resultat = resultat.sort_values('Nb_18_24_ans', ascending=False)

    print(f"  {len(resultat)} communes avec des données")

    # Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)
    print(f"Total 18-24 ans en Corse : {resultat['Nb_18_24_ans'].sum():,} personnes")
    print(f"\nTop 10 communes avec le plus de 18-24 ans:")
    print(resultat.head(10).to_string(index=False))

    print(f"\nStatistiques descriptives:")
    print(resultat['Nb_18_24_ans'].describe())

    # 5. Export CSV
    print(f"\nExport CSV vers {fichier_csv_output}...")
    resultat.to_csv(fichier_csv_output, index=False, encoding='utf-8-sig')
    print(f"  OK - {len(resultat)} communes exportées")

    # 6. Export JSON (dictionnaire code INSEE -> nombre)
    print(f"\nExport JSON vers {fichier_json_output}...")
    json_data = {
        'metadata': {
            'description': 'Nombre de personnes de 18 à 24 ans par commune en Corse',
            'source': 'INSEE - TD_POP1B_2022',
            'annee': 2022,
            'nb_communes': len(resultat),
            'total_18_24_ans': int(resultat['Nb_18_24_ans'].sum())
        },
        'data': resultat.set_index('Code_INSEE')['Nb_18_24_ans'].to_dict()
    }

    with open(fichier_json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"  OK - {len(json_data['data'])} communes exportées")

    print("\n" + "="*80)
    print("EXPORT TERMINE AVEC SUCCES")
    print("="*80)

    return resultat


if __name__ == '__main__':
    resultat = extraire_jeunes_corse()
