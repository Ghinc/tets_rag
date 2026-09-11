import pandas as pd

# Charger les données
df = pd.read_parquet(r'stage_ambroise\Données\RP2022_logemt.parquet')

# Chercher Ajaccio avec différents codes possibles
ajaccio_codes = ['2A004', '20004']

for code in ajaccio_codes:
    ajaccio = df[df['COMMUNE'] == code]
    if len(ajaccio) > 0:
        print(f'Code commune trouvé: {code}')
        print(f'Nombre de logements: {len(ajaccio)}')
        print(f'\nIRIS uniques dans cette commune:')
        iris_list = sorted(ajaccio['IRIS'].unique())
        for iris in iris_list:
            count = len(ajaccio[ajaccio['IRIS'] == iris])
            print(f'  {iris}: {count} logements')
        break
else:
    print("Ajaccio non trouvé avec les codes testés")
    print("\nPremiers codes communes dans le fichier:")
    print(df['COMMUNE'].unique()[:20])
