import json

# Charger le JSON
with open('lisa_clusters.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Créer le contenu JavaScript
js_content = """// Données LISA - Générées automatiquement (SEED=42)
// Ne pas modifier manuellement
// Correspond exactement à l'image PNG lisa_clusters_normalisation_elementaire.png

const LISA_DATA = """ + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"

# Sauvegarder
with open('lisa_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("lisa_data.js régénéré avec succès")
print(f"  {len(data['clusters'])} communes")
print(f"  Moran I: {data['metadata']['moran_global_I']:.4f}")
print(f"  Seed: {data['metadata']['seed']}")
