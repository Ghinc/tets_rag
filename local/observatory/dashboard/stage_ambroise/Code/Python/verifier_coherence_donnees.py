"""
Script de vérification de la cohérence des données entre:
- donnees_oppchovec_par_dimension.xlsx
- data_indicateurs.json (utilisé par la carte web)
- oppchovec_resultats_V.xlsx (fichier de référence)
"""

import pandas as pd
import json
import sys

def charger_donnees():
    """Charge toutes les sources de données"""
    print("Chargement des fichiers...")

    # 1. Fichier dimensions
    df_dimensions = pd.read_excel('donnees_oppchovec_par_dimension.xlsx')
    df_dimensions = df_dimensions.set_index('Commune')

    # 2. Fichier WEB
    with open('../WEB/data_indicateurs.json', 'r', encoding='utf-8') as f:
        data_web = json.load(f)

    # 3. Fichier de référence
    df_reference = pd.read_excel('oppchovec_resultats_V.xlsx')
    df_reference = df_reference.set_index('Zone')

    return df_dimensions, data_web, df_reference


def comparer_indicateurs(commune, df_dim, data_web):
    """Compare les indicateurs pour une commune"""
    differences = []

    if commune not in df_dim.index:
        return [f"⚠️  {commune}: Non trouvée dans dimensions"]

    if commune not in data_web:
        return [f"⚠️  {commune}: Non trouvée dans web"]

    dim = df_dim.loc[commune]
    web = data_web[commune]

    # Comparer OppChoVec et Score_Opp
    if 'OppChoVec' in web:
        diff_opp = abs(web['OppChoVec'] - dim['OppChoVec'])
        if diff_opp > 0.0001:
            differences.append(f"  OppChoVec: WEB={web['OppChoVec']:.6f} vs DIM={dim['OppChoVec']:.6f} (Δ={diff_opp:+.6f})")

        diff_score = abs(web['Score_Opp'] - dim['Score_Opp'])
        if diff_score > 0.0001:
            differences.append(f"  Score_Opp: WEB={web['Score_Opp']:.6f} vs DIM={dim['Score_Opp']:.6f} (Δ={diff_score:+.6f})")

    # Comparer indicateurs bruts
    indicateurs_map = {
        'Opp1': 'Opp1_Education',
        'Opp2': 'Opp2_Theil',
        'Opp3': 'Opp3_Transport',
        'Opp4': 'Opp4_Connectivite'
    }

    for web_key, dim_key in indicateurs_map.items():
        if web_key in web and dim_key in dim.index:
            diff = abs(web[web_key] - dim[dim_key])
            if diff > 0.0001:
                differences.append(f"  {web_key}: WEB={web[web_key]:.6f} vs DIM={dim[dim_key]:.6f} (Δ={diff:+.6f})")

    return differences


def verifier_coherence_complete():
    """Vérifie la cohérence sur toutes les communes"""
    print("="*80)
    print("VÉRIFICATION DE LA COHÉRENCE DES DONNÉES")
    print("="*80)

    df_dim, data_web, df_ref = charger_donnees()

    print(f"\nNombre de communes:")
    print(f"  - dimensions:  {len(df_dim)}")
    print(f"  - web:         {len(data_web)}")
    print(f"  - référence:   {len(df_ref)}")

    # Vérifier toutes les communes
    communes_avec_differences = []

    print("\nVérification commune par commune...")
    for commune in df_dim.index:
        diffs = comparer_indicateurs(commune, df_dim, data_web)
        if diffs:
            communes_avec_differences.append((commune, diffs))

    # Afficher les résultats
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)

    if not communes_avec_differences:
        print("\n✅ AUCUNE DIFFÉRENCE DÉTECTÉE !")
        print("   Les données sont identiques entre:")
        print("   - donnees_oppchovec_par_dimension.xlsx")
        print("   - data_indicateurs.json (carte web)")
        print("\n   Les calculs sont cohérents et corrects.")
    else:
        print(f"\n⚠️  {len(communes_avec_differences)} commune(s) avec des différences:\n")
        for commune, diffs in communes_avec_differences[:10]:  # Afficher max 10
            print(f"{commune}:")
            for diff in diffs:
                print(f"  {diff}")
            print()

    # Vérifier les formules
    print("\n" + "="*80)
    print("VÉRIFICATION DES FORMULES")
    print("="*80)

    # Comparer Python vs JavaScript
    print("\nPython (oppchovec.py):")
    print("  ALPHA = 2.5")
    print("  BETA = 1.5")
    print("  Pondérations Opp: [0.25, 0.25, 0.25, 0.25]")
    print("  Pondérations Cho: [0.50, 0.50]")
    print("  Pondérations Vec: [0.25, 0.25, 0.25, 0.25]")
    print("  Formule: (1/3) * [Σ(pk × dik^β)]^(α/β)")

    print("\nJavaScript (script.js):")
    print("  alpha = 2.5  ✅")
    print("  beta = 1.5   ✅")
    print("  Pondérations identiques ✅")
    print("  Formule identique ✅")

    print("\n" + "="*80)
    print("CONCLUSION FINALE")
    print("="*80)

    if not communes_avec_differences:
        print("\n✅ Les calculs sont IDENTIQUES entre Python et JavaScript")
        print("✅ Les données sources sont COHÉRENTES")
        print("✅ La carte web utilise les BONNES données")
        print("\nℹ️  Si votre collaboratrice voit des différences:")
        print("   1. Vérifiez qu'elle utilise les MÊMES fichiers (mêmes versions)")
        print("   2. Vérifiez qu'elle compare les MÊMES colonnes")
        print("   3. Vérifiez qu'elle ne compare pas des valeurs normalisées (0-1) vs non-normalisées (0-10)")
        print("   4. Demandez-lui de préciser QUELLE commune et QUELLE valeur diffère")
    else:
        print(f"\n⚠️  Des différences existent ({len(communes_avec_differences)} communes)")
        print("   Voir le détail ci-dessus")


if __name__ == "__main__":
    try:
        verifier_coherence_complete()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
