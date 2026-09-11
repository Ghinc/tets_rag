"""
Script pour extraire le nombre d'étrangers par commune en Corse
à partir du fichier INSEE TD_NAT1_2022.xlsx
Les étrangers sont caractérisés par INATC = 2
"""

import pandas as pd
import numpy as np

def extraire_etrangers_corse(fichier_input='../../Données/TD_NAT1_2022.xlsx',
                             fichier_output='etrangers_corse_2022.xlsx'):
    """
    Extrait le nombre d'étrangers pour chaque commune de Corse
    """
    print("="*80)
    print("EXTRACTION DES ÉTRANGERS EN CORSE")
    print("="*80)

    # 1. Charger le fichier Excel (en-têtes à la ligne 10)
    print(f"\nChargement du fichier {fichier_input}...")
    df = pd.read_excel(fichier_input, header=10)
    print(f"  {len(df):,} lignes chargées")
    print(f"  {len(df.columns)} colonnes")
    print(f"  Colonnes: {df.columns.tolist()[:5]}")

    # 2. Filtrer les communes corses
    print("\nFiltrage des communes corses...")
    df['CODGEO'] = df['CODGEO'].astype(str)
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()
    print(f"  {len(df_corse)} communes corses trouvées")

    # 3. Identifier toutes les colonnes avec INATC2 (étrangers)
    print("\nIdentification des colonnes 'étrangers' (INATC2)...")
    colonnes_etrangers = [col for col in df_corse.columns if 'INATC2' in str(col)]
    print(f"  {len(colonnes_etrangers)} colonnes trouvées")
    print(f"  Exemples: {colonnes_etrangers[:5]}")

    # 4. Calculer le total d'étrangers par commune
    print("\nCalcul du nombre total d'étrangers par commune...")

    # Remplacer les valeurs non numériques par 0
    for col in colonnes_etrangers:
        df_corse[col] = pd.to_numeric(df_corse[col], errors='coerce').fillna(0)

    # Sommer toutes les colonnes INATC2
    df_corse['Nb_Etrangers'] = df_corse[colonnes_etrangers].sum(axis=1)

    # 5. Créer le fichier simplifié
    resultat = df_corse[['CODGEO', 'LIBGEO', 'Nb_Etrangers']].copy()
    resultat.columns = ['Code_INSEE', 'Commune', 'Nb_Etrangers']

    # Arrondir à 2 décimales
    resultat['Nb_Etrangers'] = resultat['Nb_Etrangers'].round(2)

    # Trier par nombre d'étrangers décroissant
    resultat = resultat.sort_values('Nb_Etrangers', ascending=False).reset_index(drop=True)

    # 6. Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)
    print(f"Total étrangers en Corse : {resultat['Nb_Etrangers'].sum():,.2f} personnes")
    print(f"\nTop 10 communes avec le plus d'étrangers:")
    print(resultat.head(10).to_string(index=False))

    print(f"\nStatistiques descriptives:")
    print(resultat['Nb_Etrangers'].describe())

    # 7. Export Excel
    print(f"\nExport Excel vers {fichier_output}...")

    # Créer un writer Excel avec plusieurs feuilles
    with pd.ExcelWriter(fichier_output, engine='openpyxl') as writer:
        # Feuille 1 : Résumé par commune
        resultat.to_excel(writer, sheet_name='Etrangers_par_commune', index=False)

        # Feuille 2 : Données détaillées (toutes les colonnes pour la Corse)
        colonnes_a_garder = ['CODGEO', 'LIBGEO'] + colonnes_etrangers + ['Nb_Etrangers']
        df_detail = df_corse[colonnes_a_garder].copy()
        df_detail.to_excel(writer, sheet_name='Donnees_detaillees', index=False)

        # Feuille 3 : Métadonnées
        metadata = pd.DataFrame({
            'Information': [
                'Source',
                'Fichier',
                'Date extraction',
                'Nombre de communes',
                'Total étrangers',
                'Définition',
                'Colonnes INATC2'
            ],
            'Valeur': [
                'INSEE, RP2022 exploitation principale',
                fichier_input,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                len(resultat),
                f"{resultat['Nb_Etrangers'].sum():,.2f}",
                'INATC=2 correspond aux étrangers (non-Français)',
                f"{len(colonnes_etrangers)} colonnes de détail par âge et sexe"
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)

    print(f"  OK - Fichier créé avec {len(resultat)} communes")
    print(f"  3 feuilles : Résumé, Données détaillées, Métadonnées")

    print("\n" + "="*80)
    print("EXPORT TERMINE AVEC SUCCES")
    print("="*80)

    return resultat


if __name__ == '__main__':
    resultat = extraire_etrangers_corse()
