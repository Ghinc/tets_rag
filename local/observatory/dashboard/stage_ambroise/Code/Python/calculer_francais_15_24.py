"""
Script pour calculer la proportion de Français de 15 à 24 ans par commune en Corse
Français 15-24 = Total 15-24 - Étrangers 15-24
"""

import pandas as pd
import numpy as np

def extraire_total_15_24(fichier_pop='../../Données/TD_POP1B_2022.csv'):
    """Extrait le total des 15-24 ans depuis TD_POP1B_2022"""
    print("\n1. Extraction du TOTAL des 15-24 ans...")
    print(f"   Fichier: {fichier_pop}")

    # Charger le fichier
    df = pd.read_csv(fichier_pop, sep=';', dtype={'CODGEO': str, 'AGED100': str})

    # Filtrer la Corse
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    # Filtrer les âges 15 à 24 ans (AGED100 de '015' à '024')
    ages_15_24 = [f'{i:03d}' for i in range(15, 25)]
    df_15_24 = df_corse[df_corse['AGED100'].isin(ages_15_24)].copy()

    # Grouper par commune et sommer (tous sexes)
    total_15_24 = df_15_24.groupby(['CODGEO', 'LIBGEO'])['NB'].sum().reset_index()
    total_15_24.columns = ['Code_INSEE', 'Commune', 'Total_15_24']

    print(f"   OK - {len(total_15_24)} communes")
    print(f"   OK - Total Corse: {total_15_24['Total_15_24'].sum():,.0f} personnes de 15-24 ans")

    return total_15_24


def extraire_etrangers_15_24(fichier_nat='../../Données/TD_NAT1_2022.xlsx'):
    """Extrait les étrangers 15-24 ans depuis TD_NAT1_2022"""
    print("\n2. Extraction des ÉTRANGERS 15-24 ans...")
    print(f"   Fichier: {fichier_nat}")

    # Charger le fichier (en-têtes ligne 10)
    df = pd.read_excel(fichier_nat, header=10)

    # Filtrer la Corse
    df['CODGEO'] = df['CODGEO'].astype(str)
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    # Identifier colonnes étrangers 15-24 ans (SEULEMENT AGE415 avec INATC2)
    # AGE415 = 15-24 ans, AGE425 = 25-54 ans (à exclure !)
    colonnes_etrangers_15_24 = [col for col in df_corse.columns
                                if 'INATC2' in str(col) and 'AGE415' in str(col)]

    print(f"   OK - {len(colonnes_etrangers_15_24)} colonnes trouvees (AGE415 = 15-24 ans): {colonnes_etrangers_15_24}")

    # Convertir en numérique
    for col in colonnes_etrangers_15_24:
        df_corse[col] = pd.to_numeric(df_corse[col], errors='coerce').fillna(0)

    # Sommer les colonnes
    df_corse['Etrangers_15_24'] = df_corse[colonnes_etrangers_15_24].sum(axis=1)

    # Garder uniquement les colonnes nécessaires
    etrangers_15_24 = df_corse[['CODGEO', 'LIBGEO', 'Etrangers_15_24']].copy()
    etrangers_15_24.columns = ['Code_INSEE', 'Commune', 'Etrangers_15_24']

    print(f"   OK - {len(etrangers_15_24)} communes")
    print(f"   OK - Total Corse: {etrangers_15_24['Etrangers_15_24'].sum():,.0f} etrangers de 15-24 ans")

    return etrangers_15_24


def calculer_francais_15_24(fichier_output='proportion_francais_15_24_corse.xlsx'):
    """Calcule la proportion de Français 15-24 ans par commune en Corse"""
    print("="*80)
    print("CALCUL DE LA PROPORTION DE FRANCAIS 15-24 ANS EN CORSE")
    print("="*80)

    # 1. Extraire les totaux
    total_15_24 = extraire_total_15_24()

    # 2. Extraire les étrangers
    etrangers_15_24 = extraire_etrangers_15_24()

    # 3. Fusionner les deux DataFrames
    print("\n3. Fusion des données...")
    resultat = pd.merge(total_15_24, etrangers_15_24, on='Code_INSEE', how='left', suffixes=('', '_nat'))

    # Utiliser le nom de commune du premier fichier (plus fiable)
    if 'Commune_nat' in resultat.columns:
        resultat = resultat.drop(columns=['Commune_nat'])

    # Remplir les NaN (communes sans étrangers) par 0
    resultat['Etrangers_15_24'] = resultat['Etrangers_15_24'].fillna(0)

    # 4. Calculer les Français 15-24
    print("   Calcul: Français 15-24 = Total 15-24 - Étrangers 15-24")
    resultat['Francais_15_24'] = resultat['Total_15_24'] - resultat['Etrangers_15_24']

    # 5. Calculer les proportions
    resultat['Proportion_Francais'] = (resultat['Francais_15_24'] / resultat['Total_15_24'] * 100).round(2)
    resultat['Proportion_Etrangers'] = (resultat['Etrangers_15_24'] / resultat['Total_15_24'] * 100).round(2)

    # Arrondir les valeurs
    resultat['Total_15_24'] = resultat['Total_15_24'].round(2)
    resultat['Etrangers_15_24'] = resultat['Etrangers_15_24'].round(2)
    resultat['Francais_15_24'] = resultat['Francais_15_24'].round(2)

    # Trier par proportion de Français décroissant
    resultat = resultat.sort_values('Proportion_Francais', ascending=False).reset_index(drop=True)

    # 6. Statistiques
    print("\n" + "="*80)
    print("RESULTATS")
    print("="*80)
    print(f"Total 15-24 ans en Corse : {resultat['Total_15_24'].sum():,.0f} personnes")
    print(f"  - Francais : {resultat['Francais_15_24'].sum():,.0f} ({resultat['Francais_15_24'].sum()/resultat['Total_15_24'].sum()*100:.1f}%)")
    print(f"  - Etrangers : {resultat['Etrangers_15_24'].sum():,.0f} ({resultat['Etrangers_15_24'].sum()/resultat['Total_15_24'].sum()*100:.1f}%)")

    print(f"\nTop 10 communes avec la PLUS FORTE proportion de Francais 15-24 ans:")
    print(resultat[['Commune', 'Total_15_24', 'Francais_15_24', 'Proportion_Francais']].head(10).to_string(index=False))

    print(f"\nTop 10 communes avec la PLUS FAIBLE proportion de Francais 15-24 ans:")
    print(resultat[['Commune', 'Total_15_24', 'Francais_15_24', 'Proportion_Francais']].tail(10).to_string(index=False))

    # 7. Export Excel
    print(f"\n4. Export Excel vers {fichier_output}...")

    with pd.ExcelWriter(fichier_output, engine='openpyxl') as writer:
        # Feuille 1 : Proportion de Français 15-24 ans par commune
        resultat.to_excel(writer, sheet_name='Proportion_Francais_15_24', index=False)

        # Feuille 2 : Statistiques par département
        stats_dept = pd.DataFrame({
            'Département': ['2A (Corse-du-Sud)', '2B (Haute-Corse)', 'Total Corse'],
            'Total_15_24': [
                resultat[resultat['Code_INSEE'].str.startswith('2A')]['Total_15_24'].sum(),
                resultat[resultat['Code_INSEE'].str.startswith('2B')]['Total_15_24'].sum(),
                resultat['Total_15_24'].sum()
            ],
            'Francais_15_24': [
                resultat[resultat['Code_INSEE'].str.startswith('2A')]['Francais_15_24'].sum(),
                resultat[resultat['Code_INSEE'].str.startswith('2B')]['Francais_15_24'].sum(),
                resultat['Francais_15_24'].sum()
            ],
            'Etrangers_15_24': [
                resultat[resultat['Code_INSEE'].str.startswith('2A')]['Etrangers_15_24'].sum(),
                resultat[resultat['Code_INSEE'].str.startswith('2B')]['Etrangers_15_24'].sum(),
                resultat['Etrangers_15_24'].sum()
            ]
        })
        stats_dept['Proportion_Francais'] = (stats_dept['Francais_15_24'] / stats_dept['Total_15_24'] * 100).round(2)
        stats_dept.to_excel(writer, sheet_name='Stats_Departement', index=False)

        # Feuille 3 : Métadonnées
        metadata = pd.DataFrame({
            'Information': [
                'Description',
                'Sources',
                'Calcul nombre',
                'Formule proportion',
                'Date extraction',
                'Nombre de communes',
                'Total 15-24 ans',
                'Francais 15-24 ans',
                'Etrangers 15-24 ans',
                'Proportion moyenne Francais'
            ],
            'Valeur': [
                'Proportion de Francais de 15 a 24 ans par commune en Corse',
                'INSEE TD_POP1B_2022 (total) + TD_NAT1_2022 (etrangers)',
                'Francais 15-24 = Total 15-24 - Etrangers 15-24',
                'Proportion = (Francais 15-24 / Total 15-24) * 100',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                len(resultat),
                f"{resultat['Total_15_24'].sum():,.0f}",
                f"{resultat['Francais_15_24'].sum():,.0f}",
                f"{resultat['Etrangers_15_24'].sum():,.0f}",
                f"{resultat['Francais_15_24'].sum()/resultat['Total_15_24'].sum()*100:.1f}%"
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)

    print(f"   OK - Fichier cree avec {len(resultat)} communes")
    print(f"   OK - 3 feuilles : Proportions, Stats par departement, Metadonnees")

    print("\n" + "="*80)
    print("EXPORT TERMINE AVEC SUCCES")
    print("="*80)

    return resultat


if __name__ == '__main__':
    resultat = calculer_francais_15_24()
