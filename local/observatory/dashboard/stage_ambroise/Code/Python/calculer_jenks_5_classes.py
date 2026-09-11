"""
Calcule les seuils de Jenks avec exactement 5 classes pour toutes les variables
"""

import json
import numpy as np
from jenkspy import jenks_breaks

def calculer_gvf(data, breaks):
    """Calcule le Goodness of Variance Fit"""
    classes = np.digitize(data, breaks[1:-1])
    sdam = 0.0
    for i in range(1, len(breaks)):
        class_data = data[classes == i-1]
        if len(class_data) > 0:
            sdam += np.sum((class_data - class_data.mean()) ** 2)

    sdat = np.sum((data - data.mean()) ** 2)
    gvf = (sdat - sdam) / sdat if sdat > 0 else 0
    return gvf

def main():
    print("=" * 80)
    print("  CALCUL DES SEUILS JENKS - 5 CLASSES")
    print("=" * 80)

    # Charger les données
    print("\n[INFO] Chargement des donnees...")
    with open('../WEB/data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  [OK] {len(data)} communes chargees")

    # Extraire les valeurs sur échelle 0-10
    variables = {
        'OppChoVec_0_10': [],
        'Score_Opp_0_10': [],
        'Score_Cho_0_10': [],
        'Score_Vec_0_10': []
    }

    for commune, values in data.items():
        for var in variables.keys():
            if var in values:
                variables[var].append(values[var])

    print("\n[INFO] Statistiques des variables:")
    for var, vals in variables.items():
        arr = np.array(vals)
        print(f"  {var}: Min={arr.min():.2f}, Max={arr.max():.2f}, Moyenne={arr.mean():.2f}")

    # Calculer les seuils avec 5 classes
    print("\n[INFO] Calcul des seuils avec 5 classes...")
    resultats = {}

    n_classes = 5

    for var, vals in variables.items():
        arr = np.array(vals)
        print(f"\n  [INFO] Calcul pour {var}...")

        # Calculer les breaks pour 5 classes
        breaks = jenks_breaks(arr, n_classes=n_classes)
        gvf = calculer_gvf(arr, breaks)

        # Extraire les breaks internes (sans min et max)
        breaks_internes = breaks[1:-1]

        # S'assurer que toutes les valeurs sont <= 10
        breaks_internes = [min(b, 10.0) for b in breaks_internes]

        resultats[var] = {
            'breaks': breaks_internes,
            'gvf': float(gvf),
            'nb_classes': n_classes,
            'min': float(arr.min()),
            'max': float(arr.max())
        }

        print(f"    [OK] GVF={gvf:.6f}")
        print(f"    [OK] Breaks: {[round(b, 2) for b in breaks_internes]}")

    # Créer le fichier de sortie
    output = {
        'metadata': {
            'description': 'Seuils de Jenks avec 5 classes fixées',
            'date': '2025-11-13',
            'methode': 'Jenks Natural Breaks',
            'nb_classes': 5,
            'note': 'Tous les seuils sont garantis <= 10'
        }
    }

    # Ajouter les résultats
    for var, res in resultats.items():
        output[var] = res

    # Sauvegarder
    output_path = '../WEB/seuils_jenks_optimal_gvf.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Fichier sauvegarde: {output_path}")

    # Afficher résumé
    print("\n" + "=" * 80)
    print("RESUME DES SEUILS (5 CLASSES)")
    print("=" * 80)
    for var, res in resultats.items():
        print(f"\n{var}:")
        print(f"  Nombre de classes: {res['nb_classes']}")
        print(f"  GVF: {res['gvf']:.6f}")
        print(f"  Min: {res['min']:.2f}, Max: {res['max']:.2f}")
        print(f"  Breaks: {[round(b, 2) for b in res['breaks']]}")

        # Vérifier qu'aucun break ne dépasse 10
        if any(b > 10 for b in res['breaks']):
            print(f"  [WARN] Certains breaks dépassent 10!")
        else:
            print(f"  [OK] Tous les breaks sont <= 10")

    print("\n" + "=" * 80)
    print("[OK] CALCUL TERMINE - 5 CLASSES POUR TOUTES LES VARIABLES")
    print("=" * 80)

if __name__ == "__main__":
    main()
