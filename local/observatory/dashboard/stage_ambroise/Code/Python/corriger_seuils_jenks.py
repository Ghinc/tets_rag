"""
Crée un fichier seuils_jenks.json avec les clés compatibles
"""

import json

# Charger les seuils
with open('seuils_jenks_0_10.json', 'r') as f:
    data = json.load(f)

# Créer un nouveau format avec les deux clés
seuils = data['OppChoVec_0_10']

nouveau_format = {
    "OppChoVec": seuils,  # Pour compatibilité
    "OppChoVec_0_10": seuils,  # Avec le nom complet
    "oppchovec": seuils  # En minuscules au cas où
}

# Sauvegarder
with open('../WEB/seuils_jenks.json', 'w') as f:
    json.dump(nouveau_format, f, indent=2)

print("[OK] seuils_jenks.json cree avec les cles:")
print("  - OppChoVec")
print("  - OppChoVec_0_10")
print("  - oppchovec")
print(f"\nSeuils: {seuils}")
