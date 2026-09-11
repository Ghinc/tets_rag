import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Paramètres
input_file = r'stage_ambroise\Données\RP2022_logemt.parquet'
output_parquet = r'stage_ambroise\Données\RP2022_logemt_quartiers_selectionnes.parquet'
output_csv = r'stage_ambroise\Données\RP2022_logemt_quartiers_selectionnes.csv'

# Liste des codes IRIS à extraire
quartiers = {
    '2B0330101': 'Bastia - St Joseph',
    '2B0330201': 'Bastia - Lupino',
    '2B0330402': 'Bastia - Montesoro',
    '2B0330202': 'Bastia - St Antoine',
    '2A0040703': 'Ajaccio - Finosello',
    '2A0040701': 'Ajaccio - Les Salines',
    '2A0040901': 'Ajaccio - Les Cannes',
    '2A2470102': 'Porto-Vecchio - Pifano'
}

codes_iris = list(quartiers.keys())

print("="*70)
print("EXTRACTION DES QUARTIERS SÉLECTIONNÉS")
print("="*70)
print(f"\nQuartiers à extraire:")
for code, nom in quartiers.items():
    print(f"  - {code}: {nom}")
print()

# Lire le fichier parquet
print(f"Lecture du fichier source...")
parquet_file = pq.ParquetFile(input_file)

# Liste pour stocker les données filtrées
filtered_data = []
quartiers_trouves = {}

# Lire par batch pour économiser la mémoire
for i, batch in enumerate(parquet_file.iter_batches(batch_size=1000000)):
    df_batch = batch.to_pandas()
    # Filtrer pour tous les IRIS demandés
    df_filtered = df_batch[df_batch['IRIS'].isin(codes_iris)]

    if len(df_filtered) > 0:
        filtered_data.append(df_filtered)
        # Compter par quartier
        for code in codes_iris:
            count = len(df_filtered[df_filtered['IRIS'] == code])
            if count > 0:
                if code not in quartiers_trouves:
                    quartiers_trouves[code] = 0
                quartiers_trouves[code] += count

        print(f"Batch {i+1}: {len(df_filtered)} logements trouvés")

print("\n" + "="*70)
print("RÉSULTATS PAR QUARTIER")
print("="*70)

total_rows = 0
for code in codes_iris:
    count = quartiers_trouves.get(code, 0)
    total_rows += count
    print(f"{quartiers[code]:30} ({code}): {count:4} logements")

print("="*70)
print(f"TOTAL: {total_rows} logements")
print("="*70)

if total_rows == 0:
    print(f"\nAucune donnée trouvée pour les quartiers demandés")
else:
    # Concaténer tous les résultats
    print(f"\nConcaténation des données...")
    df_final = pd.concat(filtered_data, ignore_index=True)

    # Sauvegarder en parquet
    print(f"\nSauvegarde en Parquet: {output_parquet}")
    df_final.to_parquet(output_parquet, engine='pyarrow', index=False)

    # Sauvegarder en CSV
    print(f"Sauvegarde en CSV: {output_csv}")
    df_final.to_csv(output_csv, index=False, encoding='utf-8-sig', sep=';')

    print(f"\nFichiers créés avec succès!")
    print(f"Format: {len(df_final)} lignes x {len(df_final.columns)} colonnes")
    print(f"\nAperçu des premières lignes:")
    print(df_final.head(10))

    print(f"\nRépartition finale par IRIS dans le fichier:")
    print(df_final['IRIS'].value_counts().sort_index())
