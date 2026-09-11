"""
Calcule les seuils de Jenks pour les données avec normalisation élémentaire
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def jenks_breaks(data, n_classes=5):
    """
    Calcule les seuils de classification optimaux selon la méthode de Jenks
    """
    data_sorted = sorted([x for x in data if not np.isnan(x)])
    n = len(data_sorted)

    if n == 0:
        return []

    if n_classes >= n:
        return sorted(list(set(data_sorted)))

    # Matrices pour la programmation dynamique
    mat1 = np.zeros((n, n))
    mat2 = np.zeros((n, n_classes))

    # Initialiser la matrice de variance
    for i in range(n):
        for j in range(i, n):
            segment = data_sorted[i:j+1]
            mat1[i][j] = np.var(segment) * len(segment)

    # Initialiser la première colonne (1 classe)
    for i in range(n):
        mat2[i][0] = mat1[0][i]

    # Remplir la matrice avec programmation dynamique
    for classe in range(1, n_classes):
        for i in range(classe, n):
            min_val = float('inf')
            for k in range(classe - 1, i):
                val = mat2[k][classe - 1] + mat1[k + 1][i]
                if val < min_val:
                    min_val = val
            mat2[i][classe] = min_val

    # Retrouver les indices de rupture optimaux
    breaks_indices = [n - 1]
    current_classe = n_classes - 1

    while current_classe > 0:
        i = breaks_indices[0]
        min_val = float('inf')
        best_k = -1

        for k in range(current_classe - 1, i):
            val = mat2[k][current_classe - 1] + mat1[k + 1][i]
            if abs(val - mat2[i][current_classe]) < 1e-10:
                best_k = k
                break

        breaks_indices.insert(0, best_k)
        current_classe -= 1

    # Convertir les indices en valeurs de seuil
    breaks = [data_sorted[i] for i in breaks_indices[1:]]

    return breaks


def formater_seuils_pour_js(breaks, nom):
    """
    Formate les seuils pour utilisation dans le code JavaScript
    """
    if not breaks:
        return {
            'nom': nom,
            'n_classes': 0,
            'seuils': [],
            'ranges': []
        }

    seuils_complets = breaks

    # Créer les intervalles
    ranges = []
    for i in range(len(seuils_complets)):
        if i == 0:
            ranges.append({
                'min': 'min',
                'max': seuils_complets[i],
                'label': f'<= {seuils_complets[i]:.2f}'
            })
        else:
            ranges.append({
                'min': seuils_complets[i-1],
                'max': seuils_complets[i],
                'label': f'{seuils_complets[i-1]:.2f} - {seuils_complets[i]:.2f}'
            })

    # Dernière classe
    ranges.append({
        'min': seuils_complets[-1],
        'max': 'max',
        'label': f'> {seuils_complets[-1]:.2f}'
    })

    return {
        'nom': nom,
        'n_classes': len(ranges),
        'seuils': seuils_complets,
        'ranges': ranges
    }


def main():
    """
    Fonction principale
    """
    print("=" * 80)
    print("CALCUL DES SEUILS DE JENKS - NORMALISATION ELEMENTAIRE")
    print("=" * 80)
    print()

    # Charger les données
    print("Chargement des donnees...")
    with open('data_scores_normalisation_elem.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convertir en DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    print(f"  {len(df)} communes chargees")

    # Calculer les seuils pour chaque indicateur
    indicateurs = {
        'OppChoVec': 'OppChoVec',
        'Opp': 'Score_Opp',
        'Cho': 'Score_Cho',
        'Vec': 'Score_Vec'
    }

    resultats_jenks = {}

    print("\n" + "=" * 80)
    print("CALCUL DES SEUILS (Methode de Jenks classique)")
    print("=" * 80)

    for nom, colonne in indicateurs.items():
        print(f"\n{nom} ({colonne}):")

        valeurs = df[colonne].tolist()

        try:
            breaks = jenks_breaks(valeurs, n_classes=5)
            info = formater_seuils_pour_js(breaks, nom)
            resultats_jenks[nom] = info

            print(f"  {info['n_classes']} classes creees")
            print(f"  Seuils: {[f'{s:.3f}' for s in info['seuils']]}")
            print(f"\n  Classes:")
            for i, r in enumerate(info['ranges'], 1):
                print(f"    {i}. {r['label']}")
        except Exception as e:
            print(f"  Erreur: {e}")

    # Export
    print("\n" + "=" * 80)
    print("EXPORT")
    print("=" * 80)

    output = {
        'methode': 'jenks_classique',
        'description': 'Seuils pour donnees avec normalisation elementaire (0-1) puis OppChoVec normalise (1-10)',
        'seuils': resultats_jenks
    }

    with open('jenks_normalisation_elem.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n[OK] jenks_normalisation_elem.json")

    # Créer aussi un fichier JS prêt à l'emploi
    print("\nGeneration du fichier JavaScript...")

    js_content = "// Seuils de classification pour normalisation elementaire\n"
    js_content += "// Genere automatiquement\n\n"

    # Extraire seuils pour 4 classes (on prend les 3 premiers seuils)
    for nom, info in resultats_jenks.items():
        if len(info['seuils']) >= 3:
            seuils_4_classes = info['seuils'][:3]
            js_content += f"const seuils_{nom}_elem = {json.dumps(seuils_4_classes)};\n"

    with open('jenks_normalisation_elem.js', 'w', encoding='utf-8') as f:
        f.write(js_content)

    print("[OK] jenks_normalisation_elem.js")

    print("\n" + "=" * 80)
    print("TERMINE")
    print("=" * 80)

    print("\nUtilisez les seuils dans 'jenks_normalisation_elem.json' pour les cartes.")


if __name__ == "__main__":
    main()
