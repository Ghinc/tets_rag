import pandas as pd
import numpy as np

# ==============================================================================
# ÉTAPE 1: Créer le nouveau fichier Cho1 avec le nombre de personnes
# ==============================================================================

# Charger l'ancien fichier Cho1
df_cho1 = pd.read_excel(r'stage_ambroise\Données\Corse_Commune\Cho1.xlsx')

# Charger les données de personnes par ville
df_inper = pd.read_csv(r'stage_ambroise\Données\inper_quartiers_prioritaires_par_ville.csv', sep=';')

# Créer un mapping code_commune -> nb_personnes
mapping_personnes = dict(zip(df_inper['Code_Commune'], df_inper['Nb_Personnes_Quartiers_Prioritaires']))

print("="*70)
print("MISE À JOUR DU FICHIER CHO1")
print("="*70)

# Remplacer la colonne "Nombre de quartier cible politique de la ville"
# par le nombre de personnes
df_cho1['Nb_Personnes_Quartiers_Prioritaires'] = df_cho1['Code commune'].map(
    lambda x: mapping_personnes.get(x, 0)
)

# Afficher les changements
print("\nComparaison ancien vs nouveau:")
print("-"*70)
for _, row in df_cho1.iterrows():
    nb_quartiers = row['Nombre de quartier cible politique de la ville']
    nb_personnes = row['Nb_Personnes_Quartiers_Prioritaires']
    if nb_quartiers > 0 or nb_personnes > 0:
        ville = row['Zone']
        print(f"{ville:20} | Ancienne valeur (nb quartiers): {nb_quartiers}")
        print(f"{'':20} | Nouvelle valeur (nb personnes): {nb_personnes}")

        # Calculer les valeurs pour comparaison
        if nb_quartiers > 0:
            ancienne = np.exp(-nb_quartiers)
        else:
            ancienne = np.exp(0)  # = 1

        if nb_personnes > 0:
            nouvelle = -np.log(nb_personnes)
        else:
            nouvelle = 0

        print(f"{'':20} | exp(-{nb_quartiers}) = {ancienne:.6f}")
        print(f"{'':20} | -log({nb_personnes}) = {nouvelle:.6f}")
        print("-"*70)

# Sauvegarder le nouveau fichier
output_file = r'stage_ambroise\Données\Corse_Commune\Cho1_personnes.xlsx'
df_cho1.to_excel(output_file, index=False)

print(f"\nFichier sauvegardé: {output_file}")
print("\nColonnes du nouveau fichier:", df_cho1.columns.tolist())
print("\nNombre de communes:", len(df_cho1))

# Vérifier
print("\nVilles avec quartiers prioritaires (nb personnes > 0):")
print(df_cho1[df_cho1['Nb_Personnes_Quartiers_Prioritaires'] > 0][
    ['Zone', 'Code commune', 'Nombre de quartier cible politique de la ville', 'Nb_Personnes_Quartiers_Prioritaires']
])
