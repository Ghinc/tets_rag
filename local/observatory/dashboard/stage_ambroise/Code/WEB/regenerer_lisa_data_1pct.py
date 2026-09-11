import json

# Charger le JSON (version 1%)
with open('lisa_clusters_1pct.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Créer le contenu JavaScript
js_content = """// Données LISA - Générées automatiquement (SEED=42, alpha=1%)
// Ne pas modifier manuellement
// Correspond exactement à l'image PNG lisa_clusters_normalisation_elementaire_1pct.png

const LISA_DATA_1PCT = """ + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"

# Sauvegarder
with open('lisa_data_1pct.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("lisa_data_1pct.js régénéré avec succès")
print(f"  {len(data['clusters'])} communes")
print(f"  Moran I: {data['metadata']['moran_global_I']:.4f}")
print(f"  Seed: {data['metadata']['seed']}")
print(f"  Seuil: 1%")
