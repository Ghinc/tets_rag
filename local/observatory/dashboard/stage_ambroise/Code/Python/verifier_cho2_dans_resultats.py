"""
Script de verification que CHO2 (proportion Francais 18+) est bien utilise
dans tous les resultats oppchovec
"""

import pandas as pd
from pathlib import Path

def verifier_cho2():
    print("="*80)
    print("VERIFICATION CHO2 DANS LES RESULTATS OPPCHOVEC")
    print("="*80)
    print()

    # 1. Charger le fichier source CHO2
    cho2_file = Path("../../Données/Corse_Commune/Cho2_francais_18plus.xlsx")
    print(f"1. Chargement du fichier source: {cho2_file.name}")
    df_cho2_source = pd.read_excel(cho2_file)
    print(f"   {len(df_cho2_source)} communes")
    print(f"   Colonnes: {list(df_cho2_source.columns)}")
    print()
    print("   Exemples:")
    print(df_cho2_source[['Zone', 'Proportion_Francais_18plus']].head(5))
    print()

    # 2. Charger df_indicateur.xlsx (valeurs brutes)
    print("2. Chargement de df_indicateur.xlsx (valeurs brutes)...")
    df_indicateurs = pd.read_excel('df_indicateur.xlsx')
    print(f"   {len(df_indicateurs)} communes")
    print()
    print("   Exemples:")
    print(df_indicateurs[['Zone', 'Cho2']].head(5))
    print()

    # 3. Charger oppchovec_normalisation_elementaire.xlsx
    print("3. Chargement de oppchovec_normalisation_elementaire.xlsx...")
    df_normalise_cho = pd.read_excel('oppchovec_normalisation_elementaire.xlsx',
                                      sheet_name='Cho')
    print(f"   Sheet 'Cho': {len(df_normalise_cho)} communes")
    print()
    print("   Exemples:")
    print(df_normalise_cho[['Zone', 'Cho2']].head(5))
    print()

    # 4. Verifier la correspondance
    print("="*80)
    print("VERIFICATION DE LA CORRESPONDANCE")
    print("="*80)
    print()

    # Merge des donnees
    merged = df_cho2_source.merge(df_indicateurs[['Zone', 'Cho2']],
                                   on='Zone')
    merged.rename(columns={'Cho2': 'Cho2_indicateurs'}, inplace=True)

    merged = merged.merge(df_normalise_cho[['Zone', 'Cho2']],
                          on='Zone')
    merged.rename(columns={'Cho2': 'Cho2_normalise'}, inplace=True)

    # Calculer les differences
    merged['Diff_indicateurs'] = abs(merged['Proportion_Francais_18plus'] -
                                     merged['Cho2_indicateurs'])
    merged['Diff_normalise'] = abs(merged['Cho2_indicateurs'] -
                                   merged['Cho2_normalise'])

    print(f"Comparaison Source vs df_indicateur.xlsx:")
    print(f"  Erreur max: {merged['Diff_indicateurs'].max():.6f}")
    print(f"  Erreur moyenne: {merged['Diff_indicateurs'].mean():.6f}")
    print()

    print(f"Comparaison df_indicateur.xlsx vs normalisation_elementaire.xlsx:")
    print(f"  Erreur max: {merged['Diff_normalise'].max():.6f}")
    print(f"  Erreur moyenne: {merged['Diff_normalise'].mean():.6f}")
    print()

    # Resultat
    if merged['Diff_indicateurs'].max() < 0.01 and merged['Diff_normalise'].max() < 0.01:
        print("RESULTAT: Les valeurs CHO2 correspondent parfaitement!")
        print()
        print("Le fichier Cho2_francais_18plus.xlsx (proportion Francais 18+)")
        print("est bien utilise dans tous les calculs oppchovec.")
    else:
        print("ATTENTION: Il y a des differences detectees!")
        print()
        if merged['Diff_indicateurs'].max() >= 0.01:
            print("Communes avec differences (source vs indicateurs):")
            print(merged[merged['Diff_indicateurs'] >= 0.01][
                ['Zone', 'Proportion_Francais_18plus', 'Cho2_indicateurs',
                 'Diff_indicateurs']
            ].head(10))
        print()

    # Afficher quelques exemples de verification
    print("="*80)
    print("EXEMPLES DE VERIFICATION (10 premieres communes)")
    print("="*80)
    print()
    print(merged[['Zone', 'Proportion_Francais_18plus', 'Cho2_indicateurs',
                  'Cho2_normalise']].head(10).to_string(index=False))
    print()

    # Statistiques sur CHO2
    print("="*80)
    print("STATISTIQUES CHO2 (Proportion Francais 18+)")
    print("="*80)
    print()
    print(df_cho2_source['Proportion_Francais_18plus'].describe())
    print()
    print(f"Communes avec 100% de Francais (valeurs plafonnees): "
          f"{(df_cho2_source['Proportion_Francais_18plus'] == 100).sum()}")

    print()
    print("="*80)
    print("VERIFICATION TERMINEE")
    print("="*80)

if __name__ == "__main__":
    verifier_cho2()
