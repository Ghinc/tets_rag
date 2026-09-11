"""
Script de vérification pour diagnostiquer les problèmes d'affichage des couleurs
"""

import json
import pandas as pd


def main():
    print("=" * 80)
    print("VERIFICATION DES DONNEES POUR LA CARTE")
    print("=" * 80)
    print()

    # 1. Vérifier les seuils Jenks
    print("1. VERIFICATION DES SEUILS JENKS")
    print("-" * 80)
    with open('seuils_jenks_0_10.json', 'r', encoding='utf-8') as f:
        jenks = json.load(f)

    seuils = jenks['OppChoVec_0_10']
    print(f"  Nombre de seuils: {len(seuils)}")
    print(f"  Seuils: {[f'{s:.2f}' for s in seuils]}")

    if len(seuils) == 6:
        print("  [OK] 6 seuils présents (5 classes)")
    else:
        print(f"  [ERREUR] Attendu 6 seuils, trouvé {len(seuils)}")

    if seuils[0] == 0.0:
        print("  [OK] Minimum = 0.0")
    else:
        print(f"  [ERREUR] Minimum = {seuils[0]}, attendu 0.0")

    if abs(seuils[-1] - 10.0) < 0.01:
        print("  [OK] Maximum = 10.0")
    else:
        print(f"  [ERREUR] Maximum = {seuils[-1]}, attendu 10.0")

    # 2. Vérifier les données de scores
    print("\n2. VERIFICATION DES DONNEES DE SCORES")
    print("-" * 80)
    with open('data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Nombre de communes: {len(data)}")

    # Vérifier que toutes les communes ont OppChoVec_0_10
    communes_sans_score = []
    scores = []

    for commune, donnees in data.items():
        if 'OppChoVec_0_10' not in donnees:
            communes_sans_score.append(commune)
        else:
            scores.append(donnees['OppChoVec_0_10'])

    if communes_sans_score:
        print(f"  [ERREUR] {len(communes_sans_score)} communes sans OppChoVec_0_10:")
        for c in communes_sans_score[:5]:
            print(f"    - {c}")
    else:
        print("  [OK] Toutes les communes ont OppChoVec_0_10")

    # Vérifier les valeurs
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        print(f"  Scores OppChoVec_0_10:")
        print(f"    Min:     {min_score:.2f}")
        print(f"    Max:     {max_score:.2f}")
        print(f"    Moyenne: {sum(scores)/len(scores):.2f}")

        if min_score < 0:
            print(f"  [ERREUR] Score minimum négatif: {min_score}")
        elif min_score >= 0 and min_score <= 10:
            print(f"  [OK] Score minimum valide")

        if max_score > 10:
            print(f"  [ERREUR] Score maximum > 10: {max_score}")
        elif max_score >= 0 and max_score <= 10:
            print(f"  [OK] Score maximum valide")

    # 3. Vérifier les noms de communes
    print("\n3. VERIFICATION DES NOMS DE COMMUNES")
    print("-" * 80)

    # Exemples de communes
    exemples = ['Ajaccio', 'Bastia', 'Afa', 'Alata', 'Appietto']
    for commune in exemples:
        if commune in data:
            score = data[commune].get('OppChoVec_0_10', 'N/A')
            if isinstance(score, (int, float)):
                print(f"  [OK] {commune:20s} - Score: {score:.2f}")
            else:
                print(f"  [OK] {commune:20s} - Score: {score}")
        else:
            print(f"  [ERREUR] {commune:20s} - Non trouvée dans data_scores_0_10.json")

    # 4. Vérifier LISA
    print("\n4. VERIFICATION DES DONNEES LISA")
    print("-" * 80)

    for pct in ['1pct', '5pct']:
        fichier = f'lisa_oppchovec_0_10_{pct}.json'
        try:
            with open(fichier, 'r', encoding='utf-8') as f:
                lisa = json.load(f)
            print(f"  {fichier}:")
            print(f"    Communes: {len(lisa)}")
            significatifs = sum(1 for v in lisa.values() if v.get('significatif', False))
            print(f"    Significatifs: {significatifs} ({significatifs/len(lisa)*100:.1f}%)")

            # Compter les types de clusters
            clusters = {}
            for donnees in lisa.values():
                cluster = donnees.get('cluster', 'Inconnu')
                clusters[cluster] = clusters.get(cluster, 0) + 1

            for cluster, count in sorted(clusters.items()):
                print(f"      {cluster}: {count}")

        except FileNotFoundError:
            print(f"  [ERREUR] Fichier {fichier} non trouvé")

    # 5. Vérifier le fichier de configuration
    print("\n5. VERIFICATION DU FICHIER DE CONFIGURATION")
    print("-" * 80)

    try:
        with open('config_0_10.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("  [OK] config_0_10.json chargé")
        print(f"  Version: {config.get('version', 'N/A')}")
        print(f"  Normalisation: {config.get('normalisation', {}).get('type', 'N/A')}")

        fichiers = config.get('fichiers', {})
        print("  Fichiers référencés:")
        print(f"    - scores: {fichiers.get('scores', 'N/A')}")
        print(f"    - jenks: {fichiers.get('jenks', 'N/A')}")

    except FileNotFoundError:
        print("  [ERREUR] config_0_10.json non trouvé")

    # 6. Résumé
    print("\n" + "=" * 80)
    print("RESUME")
    print("=" * 80)
    print()
    print("Si les couleurs ne s'affichent pas sur la carte, vérifiez :")
    print()
    print("1. Que les noms de communes dans l'application web correspondent")
    print("   exactement aux noms dans data_scores_0_10.json (accents, majuscules)")
    print()
    print("2. Que le code JavaScript charge bien les fichiers:")
    print("   - data_scores_0_10.json")
    print("   - seuils_jenks_0_10.json")
    print()
    print("3. Que la fonction de colorisation utilise bien 'OppChoVec_0_10'")
    print("   et non 'OppChoVec' ou une autre colonne")
    print()
    print("4. Consultez la console du navigateur (F12) pour voir les erreurs")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
