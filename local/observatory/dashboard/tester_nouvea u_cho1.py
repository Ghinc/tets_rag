import sys
sys.path.append('stage_ambroise/Code/Python')

from oppchovec import calc_cho1, charger_donnees_commune, calculer_indicateurs
import pandas as pd

print("="*70)
print("TEST DU NOUVEAU CALCUL CHO1 AVEC -log(nb_personnes)")
print("="*70)

# Test avec les trois villes ayant des quartiers prioritaires
villes_test = {
    '2A004': 'Ajaccio',
    '2B033': 'Bastia',
    '2A247': 'Porto-Vecchio'
}

print("\nTest de la fonction calc_cho1:")
print("-"*70)

# Test direct de calc_cho1
test_valeurs = [3717, 6797, 780, 0]
for nb_pers in test_valeurs:
    resultat = calc_cho1(nb_pers)
    print(f"calc_cho1({nb_pers:5}) = {resultat:10.6f}")

print("\n" + "="*70)
print("TEST AVEC LES DONNÉES RÉELLES DES COMMUNES")
print("="*70)

# Tester le chargement et calcul pour chaque ville
for code_commune, nom_ville in villes_test.items():
    print(f"\n{nom_ville} ({code_commune}):")
    print("-"*70)

    try:
        # Charger les données de la commune
        donnees = charger_donnees_commune(code_commune)

        if 'Cho1' in donnees:
            nb_personnes = donnees['Cho1']
            print(f"  Nombre de personnes (quartiers prioritaires): {nb_personnes}")

            # Calculer Cho1
            cho1_resultat = calc_cho1(nb_personnes)
            print(f"  Cho1 = -log({nb_personnes}) = {cho1_resultat:.6f}")
        else:
            print(f"  ERREUR: Donnée Cho1 non trouvée pour {nom_ville}")

    except Exception as e:
        print(f"  ERREUR: {e}")

print("\n" + "="*70)
print("TEST TERMINÉ")
print("="*70)
