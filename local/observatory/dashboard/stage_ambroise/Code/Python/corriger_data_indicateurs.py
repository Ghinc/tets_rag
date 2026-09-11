"""
Remplace les valeurs OppChoVec par les valeurs normalisées 0-10
"""

import json

print("Correction de data_indicateurs.json...")

# Charger le fichier
with open('../WEB/data_indicateurs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Pour chaque commune, remplacer les valeurs par celles normalisées
for commune, donnees in data.items():
    # Remplacer les scores principaux par les versions 0-10
    if 'OppChoVec_0_10' in donnees:
        donnees['OppChoVec'] = donnees['OppChoVec_0_10']

    if 'Score_Opp_0_10' in donnees:
        donnees['Score_Opp'] = donnees['Score_Opp_0_10']

    if 'Score_Cho_0_10' in donnees:
        donnees['Score_Cho'] = donnees['Score_Cho_0_10']

    if 'Score_Vec_0_10' in donnees:
        donnees['Score_Vec'] = donnees['Score_Vec_0_10']

# Sauvegarder
with open('../WEB/data_indicateurs.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[OK] {len(data)} communes corrigees")
print("\nExemple Afa:")
print(f"  OppChoVec: {data['Afa']['OppChoVec']:.2f}")
print(f"  Score_Opp: {data['Afa']['Score_Opp']:.2f}")
print(f"  Score_Cho: {data['Afa']['Score_Cho']:.2f}")
print(f"  Score_Vec: {data['Afa']['Score_Vec']:.2f}")
