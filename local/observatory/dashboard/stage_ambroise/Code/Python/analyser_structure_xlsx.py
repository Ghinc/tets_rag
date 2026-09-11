import pandas as pd

# Lire sans en-têtes
df = pd.read_excel('../../Données/TD_NAT1_2022.xlsx', header=None, nrows=15)

print("Structure des 15 premières lignes:")
for i in range(15):
    print(f"\nLigne {i}:", df.iloc[i, :5].tolist())
