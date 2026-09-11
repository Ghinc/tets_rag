"""
Génère un fichier Excel avec 4 feuilles :
- Opp : Score_Opp_0_10 + sous-indicateurs Opp1, Opp2, Opp3, Opp4
- Cho : Score_Cho_0_10 + sous-indicateurs Cho1, Cho2
- Vec : Score_Vec_0_10 + sous-indicateurs Vec1, Vec2, Vec3, Vec4
- OppChoVec : OppChoVec_0_10 + les 3 scores de dimensions
"""

import pandas as pd
import json


def main():
    print("=" * 80)
    print("GENERATION DU TABLEAU EXCEL AVEC 4 FEUILLES (0-10)")
    print("=" * 80)
    print()

    # Charger les données
    print("Chargement des données...")
    with open('data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convertir en DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Zone'
    df.reset_index(inplace=True)

    print(f"  {len(df)} communes chargées")
    print(f"  Colonnes disponibles: {list(df.columns)[:15]}")

    # Créer le fichier Excel avec 4 feuilles
    fichier_sortie = 'tableau_indicateurs_0_10.xlsx'
    print(f"\nCréation du fichier {fichier_sortie}...")

    with pd.ExcelWriter(fichier_sortie, engine='openpyxl') as writer:

        # FEUILLE 1: OppChoVec
        print("  - Feuille 'OppChoVec'...")
        colonnes_oppchovec = ['Zone', 'OppChoVec_0_10', 'Score_Opp_0_10',
                              'Score_Cho_0_10', 'Score_Vec_0_10']

        # Vérifier les colonnes disponibles
        colonnes_disponibles = [col for col in colonnes_oppchovec if col in df.columns]

        df_oppchovec = df[colonnes_disponibles].copy()
        df_oppchovec = df_oppchovec.sort_values('OppChoVec_0_10', ascending=False)
        df_oppchovec.to_excel(writer, sheet_name='OppChoVec', index=False)
        print(f"    {len(df_oppchovec)} communes")

        # FEUILLE 2: Opp (Opportunités)
        print("  - Feuille 'Opp'...")
        colonnes_opp = ['Zone', 'Score_Opp_0_10', 'Opp1', 'Opp2', 'Opp3', 'Opp4']
        colonnes_opp_disponibles = [col for col in colonnes_opp if col in df.columns]

        df_opp = df[colonnes_opp_disponibles].copy()
        df_opp = df_opp.sort_values('Score_Opp_0_10', ascending=False)
        df_opp.to_excel(writer, sheet_name='Opp', index=False)
        print(f"    {len(df_opp)} communes")

        # FEUILLE 3: Cho (Choix)
        print("  - Feuille 'Cho'...")
        colonnes_cho = ['Zone', 'Score_Cho_0_10', 'Cho1', 'Cho2']
        colonnes_cho_disponibles = [col for col in colonnes_cho if col in df.columns]

        df_cho = df[colonnes_cho_disponibles].copy()
        df_cho = df_cho.sort_values('Score_Cho_0_10', ascending=False)
        df_cho.to_excel(writer, sheet_name='Cho', index=False)
        print(f"    {len(df_cho)} communes")

        # FEUILLE 4: Vec (Vécu)
        print("  - Feuille 'Vec'...")
        colonnes_vec = ['Zone', 'Score_Vec_0_10', 'Vec1', 'Vec2', 'Vec3', 'Vec4']
        colonnes_vec_disponibles = [col for col in colonnes_vec if col in df.columns]

        df_vec = df[colonnes_vec_disponibles].copy()
        df_vec = df_vec.sort_values('Score_Vec_0_10', ascending=False)
        df_vec.to_excel(writer, sheet_name='Vec', index=False)
        print(f"    {len(df_vec)} communes")

    print(f"\n[OK] Fichier créé: {fichier_sortie}")

    # Afficher les top 5 de chaque dimension
    print("\n" + "=" * 80)
    print("TOP 5 PAR DIMENSION (0-10)")
    print("=" * 80)

    print("\nOppChoVec:")
    for i, (idx, row) in enumerate(df_oppchovec.head(5).iterrows(), 1):
        print(f"  {i}. {row['Zone']:30s} - {row['OppChoVec_0_10']:.2f}/10")

    print("\nOpp (Opportunités):")
    for i, (idx, row) in enumerate(df_opp.head(5).iterrows(), 1):
        print(f"  {i}. {row['Zone']:30s} - {row['Score_Opp_0_10']:.2f}/10")

    print("\nCho (Choix):")
    for i, (idx, row) in enumerate(df_cho.head(5).iterrows(), 1):
        print(f"  {i}. {row['Zone']:30s} - {row['Score_Cho_0_10']:.2f}/10")

    print("\nVec (Vécu):")
    for i, (idx, row) in enumerate(df_vec.head(5).iterrows(), 1):
        print(f"  {i}. {row['Zone']:30s} - {row['Score_Vec_0_10']:.2f}/10")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
