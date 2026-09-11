"""
Recalcule les seuils de Jenks sur l'échelle 0-10 directement
pour éviter les valeurs > 10 dans la légende
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

def trouver_nombre_optimal_classes(data, max_classes=10, seuil_amelioration=0.01):
    """Trouve le nombre optimal de classes avec la méthode GVF"""
    meilleurs_gvf = []

    for n in range(2, max_classes + 1):
        breaks = jenks_breaks(data, n_classes=n)
        gvf = calculer_gvf(data, breaks)
        meilleurs_gvf.append({'n_classes': n, 'gvf': gvf, 'breaks': breaks})

    # Trouver le coude
    for i in range(1, len(meilleurs_gvf)):
        amelioration = meilleurs_gvf[i]['gvf'] - meilleurs_gvf[i-1]['gvf']
        if amelioration < seuil_amelioration:
            return meilleurs_gvf[i-1]

    return meilleurs_gvf[-1]

def main():
    print("=" * 80)
    print("  CORRECTION DES SEUILS JENKS - ECHELLE 0-10")
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

    # Calculer les seuils optimaux
    print("\n[INFO] Calcul des seuils optimaux avec GVF...")
    resultats = {}

    for var, vals in variables.items():
        arr = np.array(vals)
        print(f"\n  [INFO] Calcul pour {var}...")

        optimal = trouver_nombre_optimal_classes(arr, max_classes=10, seuil_amelioration=0.01)

        # Extraire les breaks internes (sans min et max)
        breaks_internes = optimal['breaks'][1:-1]

        # S'assurer que toutes les valeurs sont <= 10
        breaks_internes = [min(b, 10.0) for b in breaks_internes]

        resultats[var] = {
            'breaks': breaks_internes,
            'gvf': float(optimal['gvf']),
            'nb_classes': optimal['n_classes'],
            'min': float(arr.min()),
            'max': float(arr.max())
        }

        print(f"    [OK] {optimal['n_classes']} classes, GVF={optimal['gvf']:.6f}")
        print(f"    [OK] Breaks: {[round(b, 2) for b in breaks_internes]}")

    # Créer le fichier de sortie
    output = {
        'metadata': {
            'description': 'Seuils de Jenks calculés sur échelle 0-10 avec méthode GVF',
            'date': '2025-11-13',
            'methode': 'Jenks Natural Breaks avec optimisation GVF',
            'seuil_amelioration': 0.01,
            'note': 'Tous les seuils sont garantis <= 10'
        }
    }

    # Ajouter les résultats
    for var, res in resultats.items():
        output[var] = res

    # Calculer la recommandation globale
    nb_classes_moyennes = np.mean([res['nb_classes'] for res in resultats.values()])
    output['recommandation'] = {
        'nb_classes_global': int(np.round(nb_classes_moyennes)),
        'note': f"Moyenne arrondie des classes optimales: {nb_classes_moyennes:.1f}"
    }

    # Sauvegarder
    output_path = '../WEB/seuils_jenks_optimal_gvf.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Fichier sauvegarde: {output_path}")

    # Afficher résumé
    print("\n" + "=" * 80)
    print("RESUME DES SEUILS")
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
    print("[OK] CORRECTION TERMINEE")
    print("=" * 80)

if __name__ == "__main__":
    main()
