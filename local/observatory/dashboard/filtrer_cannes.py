import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Paramètres
input_file = r'stage_ambroise\Données\RP2022_logemt.parquet'
output_file = r'stage_ambroise\Données\RP2022_logemt_Cannes.parquet'
code_iris = '2A0040901'

print(f"Lecture du fichier source...")
print(f"Filtrage pour l'IRIS {code_iris} (Quartier des Cannes, Ajaccio)")

# Lire le fichier parquet
parquet_file = pq.ParquetFile(input_file)

# Liste pour stocker les données filtrées
filtered_data = []

# Lire par batch pour économiser la mémoire
total_rows = 0
for i, batch in enumerate(parquet_file.iter_batches(batch_size=1000000)):
    df_batch = batch.to_pandas()
    # Filtrer pour l'IRIS des Cannes
    df_filtered = df_batch[df_batch['IRIS'] == code_iris]

    if len(df_filtered) > 0:
        filtered_data.append(df_filtered)
        total_rows += len(df_filtered)
        print(f"Batch {i+1}: {len(df_filtered)} logements trouvés (Total: {total_rows})")

if total_rows == 0:
    print(f"\nAucune donnée trouvée pour l'IRIS {code_iris}")
else:
    # Concaténer tous les résultats
    print(f"\nConcaténation des données...")
    df_final = pd.concat(filtered_data, ignore_index=True)

    print(f"Nombre total de logements: {len(df_final)}")
    print(f"\nSauvegarde dans: {output_file}")

    # Sauvegarder en parquet
    df_final.to_parquet(output_file, engine='pyarrow', index=False)

    print(f"\nFichier créé avec succès!")
    print(f"Taille: {len(df_final)} lignes x {len(df_final.columns)} colonnes")
    print(f"\nAperçu des premières lignes:")
    print(df_final.head())
