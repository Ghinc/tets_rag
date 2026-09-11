import pandas as pd
import json

# Charger les résultats OppChoVec
df = pd.read_excel('oppchovec_resultats_V.xlsx')
print('Colonnes disponibles:', df.columns.tolist())

# Convertir en dictionnaire
data_dict = df.to_dict(orient='index')

# Sauvegarder en JSON
with open('data_scores.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)

print(f'Fichier data_scores.json cree avec {len(data_dict)} communes')
