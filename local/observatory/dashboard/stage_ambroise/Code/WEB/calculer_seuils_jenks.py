"""
Calcule les seuils de classification optimaux selon la méthode de Jenks (Natural Breaks)
pour les cartes OppChoVec, Opp, Cho et Vec.

La méthode de Jenks minimise la variance intra-classe et maximise la variance inter-classe.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple


def jenks_breaks(data: List[float], n_classes: int = 5) -> List[float]:
    """
    Calcule les seuils de classification optimaux selon la méthode de Jenks.

    Args:
        data: Liste des valeurs à classifier
        n_classes: Nombre de classes souhaitées

    Returns:
        Liste des seuils (n_classes - 1 valeurs)
    """
    # Trier les données
    data_sorted = sorted([x for x in data if not np.isnan(x)])
    n = len(data_sorted)

    if n == 0:
        return []

    if n_classes >= n:
        # Si on a plus de classes que de valeurs, retourner les valeurs uniques
        return sorted(list(set(data_sorted)))

    # Matrices pour la programmation dynamique
    # mat1[i][j] = variance intra-classe pour les éléments de i à j
    mat1 = np.zeros((n, n))

    # mat2[i][j] = variance cumulée optimale pour j classes et i éléments
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


def calculer_seuils_jenks_simplifies(data: List[float], n_classes: int = 5) -> List[float]:
    """
    Version simplifiée de Jenks basée sur les différences entre valeurs consécutives.
    Trouve les plus grandes différences pour déterminer les ruptures naturelles.

    Args:
        data: Liste des valeurs à classifier
        n_classes: Nombre de classes souhaitées

    Returns:
        Liste des seuils (n_classes - 1 valeurs)
    """
    # Filtrer les NaN et trier
    data_clean = sorted([x for x in data if not np.isnan(x)])

    if len(data_clean) < n_classes:
        return data_clean

    # Calculer les différences entre valeurs consécutives
    diffs = []
    for i in range(1, len(data_clean)):
        diff = data_clean[i] - data_clean[i-1]
        diffs.append((diff, i))

    # Trier par différence décroissante
    diffs_sorted = sorted(diffs, key=lambda x: x[0], reverse=True)

    # Prendre les n_classes-1 plus grandes différences
    break_indices = sorted([idx for _, idx in diffs_sorted[:n_classes-1]])

    # Convertir en seuils
    breaks = [data_clean[i] for i in break_indices]

    return breaks


def formater_seuils_pour_js(breaks: List[float], nom: str) -> dict:
    """
    Formate les seuils pour utilisation dans le code JavaScript.

    Args:
        breaks: Liste des seuils
        nom: Nom de l'indicateur

    Returns:
        Dictionnaire avec les informations de classification
    """
    if not breaks:
        return {
            'nom': nom,
            'n_classes': 0,
            'seuils': [],
            'ranges': []
        }

    # Ajouter min et max
    seuils_complets = breaks

    # Créer les intervalles
    ranges = []
    for i in range(len(seuils_complets)):
        if i == 0:
            ranges.append({
                'min': 'min',
                'max': seuils_complets[i],
                'label': f'<= {seuils_complets[i]:.3f}'
            })
        else:
            ranges.append({
                'min': seuils_complets[i-1],
                'max': seuils_complets[i],
                'label': f'{seuils_complets[i-1]:.3f} - {seuils_complets[i]:.3f}'
            })

    # Dernière classe
    ranges.append({
        'min': seuils_complets[-1],
        'max': 'max',
        'label': f'> {seuils_complets[-1]:.3f}'
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
    print("="*80)
    print("CALCUL DES SEUILS DE JENKS POUR LES CARTES")
    print("="*80)

    # Charger les données
    print("\nChargement des données...")
    with open('data_scores.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convertir en DataFrame (communes en lignes)
    df = pd.DataFrame.from_dict(data, orient='index')

    print(f"  {len(df)} communes chargées")

    # Calculer les seuils pour chaque indicateur
    indicateurs = {
        'OppChoVec': 'OppChoVec',
        'Opp': 'Score_Opp',
        'Cho': 'Score_Cho',
        'Vec': 'Score_Vec'
    }

    resultats = {}

    print("\n" + "="*80)
    print("CALCUL DES SEUILS (Méthode simplifiée basée sur les différences)")
    print("="*80)

    for nom, colonne in indicateurs.items():
        print(f"\n{nom} ({colonne}):")

        # Extraire les valeurs
        valeurs = df[colonne].tolist()

        # Calculer les seuils (5 classes)
        breaks = calculer_seuils_jenks_simplifies(valeurs, n_classes=5)

        # Formater pour JS
        info = formater_seuils_pour_js(breaks, nom)
        resultats[nom] = info

        print(f"  {info['n_classes']} classes créées")
        print(f"  Seuils: {[f'{s:.3f}' for s in info['seuils']]}")
        print(f"\n  Classes:")
        for i, r in enumerate(info['ranges'], 1):
            print(f"    {i}. {r['label']}")

    # Aussi calculer avec la vraie méthode de Jenks (pour comparaison)
    print("\n" + "="*80)
    print("CALCUL DES SEUILS (Méthode de Jenks classique)")
    print("="*80)

    resultats_jenks = {}

    for nom, colonne in indicateurs.items():
        print(f"\n{nom} ({colonne}):")

        valeurs = df[colonne].tolist()

        try:
            breaks = jenks_breaks(valeurs, n_classes=5)
            info = formater_seuils_pour_js(breaks, nom)
            resultats_jenks[nom] = info

            print(f"  {info['n_classes']} classes créées")
            print(f"  Seuils: {[f'{s:.3f}' for s in info['seuils']]}")
        except Exception as e:
            print(f"  Erreur: {e}")
            resultats_jenks[nom] = resultats[nom]  # Utiliser la version simplifiée

    # Export
    print("\n" + "="*80)
    print("EXPORT")
    print("="*80)

    output = {
        'methode_simplifiee': resultats,
        'methode_jenks_classique': resultats_jenks,
        'recommandation': 'methode_jenks_classique'
    }

    with open('seuils_jenks.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n[OK] seuils_jenks.json")

    # Créer aussi un fichier JS prêt à l'emploi
    print("\nGénération du fichier JavaScript...")

    js_content = "// Seuils de classification optimaux (méthode de Jenks)\n"
    js_content += "// Généré automatiquement par calculer_seuils_jenks.py\n\n"

    for nom, info in resultats_jenks.items():
        js_content += f"const seuils_{nom} = {json.dumps(info['seuils'], indent=2)};\n"

    js_content += "\n// Fonction pour obtenir la couleur selon la valeur\n"
    js_content += "function getColor_OppChoVec(value) {\n"
    js_content += f"  const seuils = {json.dumps(resultats_jenks['OppChoVec']['seuils'])};\n"
    js_content += "  const colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'];\n"
    js_content += "  \n"
    js_content += "  for (let i = 0; i < seuils.length; i++) {\n"
    js_content += "    if (value <= seuils[i]) return colors[i];\n"
    js_content += "  }\n"
    js_content += "  return colors[colors.length - 1];\n"
    js_content += "}\n"

    with open('seuils_jenks.js', 'w', encoding='utf-8') as f:
        f.write(js_content)

    print("[OK] seuils_jenks.js")

    print("\n" + "="*80)
    print("TERMINÉ")
    print("="*80)

    print("\nUtilisez les seuils dans 'seuils_jenks.json' pour mettre à jour les cartes.")
    print("Méthode recommandée: Jenks classique (variance minimale intra-classe)")


if __name__ == "__main__":
    main()
