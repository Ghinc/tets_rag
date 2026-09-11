import pandas as pd
import numpy as np

# Lire le fichier des quartiers sélectionnés
df = pd.read_csv(r'stage_ambroise\Données\RP2022_logemt_quartiers_selectionnes.csv', sep=';')

print("="*70)
print("CALCUL DU NOMBRE DE PERSONNES PAR VILLE (QUARTIERS PRIORITAIRES)")
print("="*70)

# Mapper les codes communes vers les noms de villes
villes = {
    '2A004': 'Ajaccio',
    '2B033': 'Bastia',
    '2A247': 'Porto-Vecchio'
}

# Convertir INPER en numérique (gérer les valeurs non numériques)
df['INPER_num'] = pd.to_numeric(df['INPER'], errors='coerce')

# Vérifier s'il y a des valeurs manquantes
nb_na = df['INPER_num'].isna().sum()
if nb_na > 0:
    print(f"\nAttention: {nb_na} valeurs INPER non numériques trouvées (remplacées par 0)")
    df['INPER_num'] = df['INPER_num'].fillna(0)

# Grouper par commune et sommer INPER
somme_par_ville = df.groupby('COMMUNE')['INPER_num'].sum().to_dict()

print("\nNombre total de personnes par ville dans les quartiers prioritaires:")
print("-" * 70)

total_personnes = 0
for code_commune, nom_ville in villes.items():
    nb_personnes = somme_par_ville.get(code_commune, 0)
    total_personnes += nb_personnes

    # Calculer -log(x)
    if nb_personnes > 0:
        valeur_log = -np.log(nb_personnes)
    else:
        valeur_log = 0

    print(f"{nom_ville:20} ({code_commune}): {nb_personnes:6.0f} personnes  -->  -log(x) = {valeur_log:8.4f}")

print("-" * 70)
print(f"{'TOTAL':20}          {total_personnes:6.0f} personnes")
print("="*70)

# Créer un DataFrame avec les résultats pour export
resultat = []
for code_commune, nom_ville in villes.items():
    nb_personnes = somme_par_ville.get(code_commune, 0)
    if nb_personnes > 0:
        valeur_log = -np.log(nb_personnes)
    else:
        valeur_log = 0

    resultat.append({
        'Code_Commune': code_commune,
        'Ville': nom_ville,
        'Nb_Personnes_Quartiers_Prioritaires': int(nb_personnes),
        'Valeur_moins_log': valeur_log
    })

df_resultat = pd.DataFrame(resultat)

# Sauvegarder
output_file = r'stage_ambroise\Données\inper_quartiers_prioritaires_par_ville.csv'
df_resultat.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')

print(f"\nFichier sauvegardé: {output_file}")
print("\nContenu du fichier:")
print(df_resultat.to_string(index=False))
