import pandas as pd
import pyarrow.parquet as pq

# Lire avec pyarrow pour voir les codes communes disponibles
parquet_file = pq.ParquetFile(r'stage_ambroise\Données\RP2022_logemt.parquet')

# Lire un échantillon pour voir les codes
df_sample = parquet_file.read_row_group(0).to_pandas()
print("Échantillon de codes communes:")
print(df_sample['COMMUNE'].unique()[:30])

# Chercher Ajaccio
print("\nRecherche d'Ajaccio...")
ajaccio_codes = ['2A004', '20004', '2A 004']

# Lire par chunks pour économiser la mémoire
for batch in parquet_file.iter_batches(batch_size=1000000):
    df_batch = batch.to_pandas()
    for code in ajaccio_codes:
        ajaccio = df_batch[df_batch['COMMUNE'] == code]
        if len(ajaccio) > 0:
            print(f'\nCode commune trouvé: {code}')
            print(f'Échantillon trouvé: {len(ajaccio)} logements dans ce batch')
            print(f'IRIS dans cet échantillon: {ajaccio["IRIS"].unique()}')
            break
    else:
        continue
    break
