import pandas as pd

# Convertir le fichier des Cannes en CSV
input_file = r'stage_ambroise\Données\RP2022_logemt_Cannes.parquet'
output_file = r'stage_ambroise\Données\RP2022_logemt_Cannes.csv'

print(f"Lecture du fichier parquet...")
df = pd.read_parquet(input_file)

print(f"Sauvegarde en CSV...")
df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')

print(f"\nFichier créé avec succès!")
print(f"Fichier: {output_file}")
print(f"Lignes: {len(df)}")
print(f"Colonnes: {len(df.columns)}")
