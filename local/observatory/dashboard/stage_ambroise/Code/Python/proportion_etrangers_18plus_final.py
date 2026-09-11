"""
NOM DU SCRIPT : proportion_etrangers_18plus_final.py

Calcul de la proportion d'étrangers de 18+ ans par commune en Corse

FORMULE EXACTE utilisée :
  Étrangers(18-24) = Total(18-24) - Français(18-24)

Ensuite :
  Étrangers 18+ = Étrangers(18-24) + Étrangers(25-54) + Étrangers(55+)
  Proportion = (Étrangers 18+ / Population totale) * 100
"""

import pandas as pd

def extraire_population_totale(fichier_pop='../../Données/TD_POP1B_2022.csv'):
    """Extrait la population totale par commune"""
    print("\n1. POPULATION TOTALE par commune")
    print("   Source: TD_POP1B_2022.csv")

    df = pd.read_csv(fichier_pop, sep=';', dtype={'CODGEO': str, 'AGED100': str})
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    pop_totale = df_corse.groupby(['CODGEO', 'LIBGEO'])['NB'].sum().reset_index()
    pop_totale.columns = ['Code_INSEE', 'Commune', 'Population_Totale']

    print(f"   --> Total Corse: {pop_totale['Population_Totale'].sum():,.0f} habitants")
    return pop_totale


def extraire_total_18_24(fichier_pop='../../Données/TD_POP1B_2022.csv'):
    """Extrait Total(18-24) depuis TD_POP1B_2022"""
    print("\n2. TOTAL(18-24) par commune")
    print("   Source: TD_POP1B_2022.csv, ages 018 a 024")

    df = pd.read_csv(fichier_pop, sep=';', dtype={'CODGEO': str, 'AGED100': str})
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    ages_18_24 = [f'{i:03d}' for i in range(18, 25)]
    df_18_24 = df_corse[df_corse['AGED100'].isin(ages_18_24)].copy()

    total_18_24 = df_18_24.groupby(['CODGEO', 'LIBGEO'])['NB'].sum().reset_index()
    total_18_24.columns = ['Code_INSEE', 'Commune', 'Total_18_24']

    print(f"   --> Total Corse 18-24: {total_18_24['Total_18_24'].sum():,.0f} personnes")
    return total_18_24


def extraire_francais_18_24(fichier_nat='../../Données/TD_NAT1_2022.xlsx'):
    """Extrait Français(18-24) depuis TD_NAT1_2022 avec INATC1"""
    print("\n3. FRANCAIS(18-24) par commune")
    print("   Source: TD_NAT1_2022.xlsx, INATC1 (Francais) AGE415 (15-24)")
    print("   Note: AGE415 = 15-24 ans, on approxime 18-24 en faisant * 7/10")

    df = pd.read_excel(fichier_nat, header=10)
    df['CODGEO'] = df['CODGEO'].astype(str)
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    # Colonnes avec INATC1 (Français) et AGE415 (15-24 ans)
    colonnes_francais_15_24 = [col for col in df_corse.columns
                               if 'INATC1' in str(col) and 'AGE415' in str(col)]

    print(f"   --> Colonnes trouvees: {colonnes_francais_15_24}")

    for col in colonnes_francais_15_24:
        df_corse[col] = pd.to_numeric(df_corse[col], errors='coerce').fillna(0)

    df_corse['Francais_15_24'] = df_corse[colonnes_francais_15_24].sum(axis=1)

    # Approximation : Français(18-24) = Français(15-24) * 7/10
    df_corse['Francais_18_24'] = df_corse['Francais_15_24'] * (7/10)

    francais_18_24 = df_corse[['CODGEO', 'LIBGEO', 'Francais_18_24']].copy()
    francais_18_24.columns = ['Code_INSEE', 'Commune', 'Francais_18_24']

    print(f"   --> Total Corse Francais 18-24: {francais_18_24['Francais_18_24'].sum():,.0f}")
    return francais_18_24


def calculer_etrangers_18_24(total_18_24, francais_18_24):
    """FORMULE : Étrangers(18-24) = Total(18-24) - Français(18-24)"""
    print("\n4. ETRANGERS(18-24) par commune")
    print("   FORMULE: Etrangers(18-24) = Total(18-24) - Francais(18-24)")

    merged = pd.merge(total_18_24, francais_18_24, on='Code_INSEE', how='left', suffixes=('', '_nat'))
    if 'Commune_nat' in merged.columns:
        merged = merged.drop(columns=['Commune_nat'])

    merged['Francais_18_24'] = merged['Francais_18_24'].fillna(0)

    # CALCUL EXACT
    merged['Etrangers_18_24'] = merged['Total_18_24'] - merged['Francais_18_24']

    print(f"   --> Total Corse Etrangers 18-24: {merged['Etrangers_18_24'].sum():,.0f}")

    return merged[['Code_INSEE', 'Commune', 'Etrangers_18_24']].copy()


def extraire_etrangers_25plus(fichier_nat='../../Données/TD_NAT1_2022.xlsx'):
    """Extrait Étrangers(25+) depuis TD_NAT1_2022"""
    print("\n5. ETRANGERS(25+) par commune")
    print("   Source: TD_NAT1_2022.xlsx, INATC2 (Etrangers) AGE425 + AGE455")

    df = pd.read_excel(fichier_nat, header=10)
    df['CODGEO'] = df['CODGEO'].astype(str)
    df_corse = df[df['CODGEO'].str.startswith(('2A', '2B'))].copy()

    # AGE425 (25-54) et AGE455 (55+) avec INATC2
    colonnes_etrangers_25plus = [col for col in df_corse.columns
                                 if 'INATC2' in str(col) and ('AGE425' in str(col) or 'AGE455' in str(col))]

    for col in colonnes_etrangers_25plus:
        df_corse[col] = pd.to_numeric(df_corse[col], errors='coerce').fillna(0)

    df_corse['Etrangers_25plus'] = df_corse[colonnes_etrangers_25plus].sum(axis=1)

    etrangers_25plus = df_corse[['CODGEO', 'LIBGEO', 'Etrangers_25plus']].copy()
    etrangers_25plus.columns = ['Code_INSEE', 'Commune', 'Etrangers_25plus']

    print(f"   --> Total Corse Etrangers 25+: {etrangers_25plus['Etrangers_25plus'].sum():,.0f}")
    return etrangers_25plus


def calculer_proportion_etrangers_18plus(fichier_output='proportion_etrangers_18plus_corse.xlsx'):
    """Calcule la proportion d'étrangers 18+ par commune"""
    print("="*80)
    print("CALCUL PROPORTION ETRANGERS 18+ ANS - CORSE")
    print("Script: proportion_etrangers_18plus_final.py")
    print("="*80)

    # Étapes de calcul
    pop_totale = extraire_population_totale()
    total_18_24 = extraire_total_18_24()
    francais_18_24 = extraire_francais_18_24()
    etrangers_18_24 = calculer_etrangers_18_24(total_18_24, francais_18_24)
    etrangers_25plus = extraire_etrangers_25plus()

    # Fusion
    print("\n6. FUSION et calcul final")
    resultat = pop_totale.copy()
    resultat = pd.merge(resultat, etrangers_18_24[['Code_INSEE', 'Etrangers_18_24']], on='Code_INSEE', how='left')
    resultat = pd.merge(resultat, etrangers_25plus[['Code_INSEE', 'Etrangers_25plus']], on='Code_INSEE', how='left')

    resultat['Etrangers_18_24'] = resultat['Etrangers_18_24'].fillna(0)
    resultat['Etrangers_25plus'] = resultat['Etrangers_25plus'].fillna(0)

    # Total étrangers 18+
    resultat['Etrangers_18plus'] = resultat['Etrangers_18_24'] + resultat['Etrangers_25plus']

    # Proportion
    resultat['Proportion_Etrangers_18plus'] = (resultat['Etrangers_18plus'] / resultat['Population_Totale'] * 100).round(2)

    # Arrondir
    resultat['Population_Totale'] = resultat['Population_Totale'].round(2)
    resultat['Etrangers_18_24'] = resultat['Etrangers_18_24'].round(2)
    resultat['Etrangers_25plus'] = resultat['Etrangers_25plus'].round(2)
    resultat['Etrangers_18plus'] = resultat['Etrangers_18plus'].round(2)

    # Trier
    resultat = resultat.sort_values('Proportion_Etrangers_18plus', ascending=False).reset_index(drop=True)

    # Statistiques
    print("\n" + "="*80)
    print("RESULTATS FINAUX")
    print("="*80)
    print(f"Population totale Corse : {resultat['Population_Totale'].sum():,.0f} habitants")
    print(f"Etrangers 18-24 ans : {resultat['Etrangers_18_24'].sum():,.0f}")
    print(f"Etrangers 25+ ans : {resultat['Etrangers_25plus'].sum():,.0f}")
    print(f"Etrangers 18+ TOTAL : {resultat['Etrangers_18plus'].sum():,.0f}")
    print(f"Proportion moyenne : {resultat['Etrangers_18plus'].sum()/resultat['Population_Totale'].sum()*100:.2f}%")

    print(f"\nTop 10 communes - PLUS FORTE proportion:")
    print(resultat[['Commune', 'Population_Totale', 'Etrangers_18plus', 'Proportion_Etrangers_18plus']].head(10).to_string(index=False))

    # Export Excel
    print(f"\n7. EXPORT vers {fichier_output}")

    with pd.ExcelWriter(fichier_output, engine='openpyxl') as writer:
        resultat.to_excel(writer, sheet_name='Proportion_Etrangers_18plus', index=False)

        stats_dept = pd.DataFrame({
            'Departement': ['2A (Corse-du-Sud)', '2B (Haute-Corse)', 'Total Corse'],
            'Population_Totale': [
                resultat[resultat['Code_INSEE'].str.startswith('2A')]['Population_Totale'].sum(),
                resultat[resultat['Code_INSEE'].str.startswith('2B')]['Population_Totale'].sum(),
                resultat['Population_Totale'].sum()
            ],
            'Etrangers_18plus': [
                resultat[resultat['Code_INSEE'].str.startswith('2A')]['Etrangers_18plus'].sum(),
                resultat[resultat['Code_INSEE'].str.startswith('2B')]['Etrangers_18plus'].sum(),
                resultat['Etrangers_18plus'].sum()
            ]
        })
        stats_dept['Proportion_Etrangers_18plus'] = (stats_dept['Etrangers_18plus'] / stats_dept['Population_Totale'] * 100).round(2)
        stats_dept.to_excel(writer, sheet_name='Stats_Departement', index=False)

        metadata = pd.DataFrame({
            'Information': [
                'Script',
                'Description',
                'Formule cle',
                'Etape 1',
                'Etape 2',
                'Etape 3',
                'Sources',
                'Date',
                'Communes',
                'Population',
                'Etrangers 18+',
                'Proportion'
            ],
            'Valeur': [
                'proportion_etrangers_18plus_final.py',
                'Proportion d\'etrangers 18+ ans par commune en Corse',
                'Etrangers(18-24) = Total(18-24) - Francais(18-24)',
                'Francais(18-24) = Francais(15-24) * 7/10 (approximation AGE415)',
                'Etrangers 18+ = Etrangers(18-24) + Etrangers(25-54) + Etrangers(55+)',
                'Proportion = (Etrangers 18+ / Population totale) * 100',
                'INSEE TD_POP1B_2022 + TD_NAT1_2022',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                len(resultat),
                f"{resultat['Population_Totale'].sum():,.0f}",
                f"{resultat['Etrangers_18plus'].sum():,.0f}",
                f"{resultat['Etrangers_18plus'].sum()/resultat['Population_Totale'].sum()*100:.2f}%"
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)

    print(f"   OK - {len(resultat)} communes exportees")
    print("\n" + "="*80)
    print("TERMINE")
    print("="*80)

    return resultat


if __name__ == '__main__':
    resultat = calculer_proportion_etrangers_18plus()
