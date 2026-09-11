"""
Génère le fichier cah_data.js avec les données CAH en tant que constante JavaScript
(similaire à lisa_data.js)
"""

import json

def generer_cah_data_js():
    print("[INFO] Génération du fichier cah_data.js...")

    # Charger le JSON
    with open('../WEB/cah_3_clusters.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Générer le contenu JavaScript
    js_content = """// Données CAH (Classification Hiérarchique Ascendante) - 3 clusters
// Généré automatiquement depuis cah_3_clusters.json
// Ne pas modifier manuellement

const CAH_DATA = """ + json.dumps(data, indent=2, ensure_ascii=False) + """;
"""

    # Sauvegarder
    output_path = '../WEB/cah_data.js'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(js_content)

    print(f"  [OK] Fichier JavaScript généré: {output_path}")
    print(f"  [INFO] Nombre de communes: {len(data['clusters'])}")
    print(f"  [INFO] Nombre de clusters: {data['metadata']['n_clusters']}")

if __name__ == "__main__":
    print("=" * 80)
    print("  GÉNÉRATION DU FICHIER cah_data.js")
    print("=" * 80)
    generer_cah_data_js()
    print("=" * 80)
