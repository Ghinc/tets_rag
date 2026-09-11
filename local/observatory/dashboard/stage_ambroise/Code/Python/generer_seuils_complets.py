"""
Génère les seuils Jenks pour TOUTES les dimensions normalisées 0-10
"""

import pandas as pd
import numpy as np
import json


def jenks_breaks(values, n_classes=5):
    """
    Calcule les seuils de Jenks (Natural Breaks)
    """
    values = np.array(sorted(values))
    n = len(values)

    if n <= n_classes:
        return list(values)

    # Matrices pour la programmation dynamique
    mat1 = np.zeros((n + 1, n_classes + 1))
    mat2 = np.zeros((n + 1, n_classes + 1))

    # Initialisation
    for i in range(1, n_classes + 1):
        mat1[1, i] = 1
        mat2[1, i] = 0
        for j in range(2, n + 1):
            mat2[j, i] = float('inf')

    # Calcul
    for l in range(2, n + 1):
        sum_val = 0
        sum_sq = 0
        w = 0

        for m in range(1, l + 1):
            i3 = l - m + 1
            val = values[i3 - 1]

            sum_val += val
            sum_sq += val * val
            w += 1

            variance = sum_sq - (sum_val * sum_val) / w

            if i3 != 1:
                for j in range(2, n_classes + 1):
                    if mat2[l, j] >= variance + mat2[i3 - 1, j - 1]:
                        mat1[l, j] = i3
                        mat2[l, j] = variance + mat2[i3 - 1, j - 1]

        mat1[l, 1] = 1
        mat2[l, 1] = variance

    # Extraction des seuils
    breaks = []
    k = n
    for j in range(n_classes, 0, -1):
        breaks.insert(0, values[int(mat1[k, j]) - 1])
        k = int(mat1[k, j]) - 1

    # Ajouter le max
    breaks.append(values[-1])

    return breaks


def main():
    print("=" * 80)
    print("GENERATION DES SEUILS JENKS COMPLETS (0-10)")
    print("=" * 80)
    print()

    # Charger les données
    print("Chargement de data_scores_0_10.json...")
    with open('data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    print(f"  {len(df)} communes chargées")

    # Variables à traiter
    variables = {
        'OppChoVec_0_10': 'OppChoVec (0-10)',
        'Score_Opp_0_10': 'Opportunités (0-10)',
        'Score_Cho_0_10': 'Choix (0-10)',
        'Score_Vec_0_10': 'Vécu (0-10)'
    }

    # Calculer les seuils Jenks pour chaque variable
    print("\nCalcul des seuils Jenks (5 classes)...")
    print("-" * 80)

    seuils_jenks = {}

    for var, description in variables.items():
        if var in df.columns:
            values = df[var].dropna().values
            breaks = jenks_breaks(values, n_classes=5)
            seuils_jenks[var] = breaks

            print(f"\n{description}:")
            print(f"  Variable: {var}")
            print(f"  Seuils: {[f'{b:.2f}' for b in breaks]}")
            print(f"  Classes:")
            for i in range(len(breaks)-1):
                print(f"    Classe {i+1}: {breaks[i]:.2f} - {breaks[i+1]:.2f}")

    # Ajouter des alias pour compatibilité
    seuils_jenks['OppChoVec'] = seuils_jenks['OppChoVec_0_10']
    seuils_jenks['Score_Opp'] = seuils_jenks['Score_Opp_0_10']
    seuils_jenks['Score_Cho'] = seuils_jenks['Score_Cho_0_10']
    seuils_jenks['Score_Vec'] = seuils_jenks['Score_Vec_0_10']

    # Sauvegarder
    print("\n" + "=" * 80)
    print("SAUVEGARDE")
    print("=" * 80)

    fichier_sortie = 'seuils_jenks_complet.json'
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        json.dump(seuils_jenks, f, indent=2, ensure_ascii=False)

    print(f"[OK] {fichier_sortie}")
    print(f"\nClés disponibles: {list(seuils_jenks.keys())}")

    # Copier vers WEB
    import shutil
    shutil.copy(fichier_sortie, '../WEB/seuils_jenks.json')
    print(f"[OK] Copié vers ../WEB/seuils_jenks.json")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
