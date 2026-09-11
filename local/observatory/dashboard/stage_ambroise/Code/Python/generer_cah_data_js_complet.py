"""
Génère le fichier cah_data.js avec les données CAH (3 et 5 clusters) en tant que constantes JavaScript
"""

import json

def generer_cah_data_js():
    print("[INFO] Génération du fichier cah_data.js...")

    # Charger les JSON
    with open('../WEB/cah_3_clusters.json', 'r', encoding='utf-8') as f:
        data_3 = json.load(f)

    with open('../WEB/cah_5_clusters.json', 'r', encoding='utf-8') as f:
        data_5 = json.load(f)

    # Générer le contenu JavaScript
    js_content = """// Données CAH (Classification Hiérarchique Ascendante)
// Généré automatiquement depuis cah_3_clusters.json et cah_5_clusters.json
// Ne pas modifier manuellement

// Données pour 3 clusters
const CAH_DATA_3 = """ + json.dumps(data_3, indent=2, ensure_ascii=False) + """;

// Données pour 5 clusters
const CAH_DATA_5 = """ + json.dumps(data_5, indent=2, ensure_ascii=False) + """;

// Par défaut, utiliser CAH_DATA_3 (compatibilité avec ancien code)
const CAH_DATA = CAH_DATA_3;
"""

    # Sauvegarder
    output_path = '../WEB/cah_data.js'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(js_content)

    print(f"  [OK] Fichier JavaScript généré: {output_path}")
    print(f"  [INFO] Nombre de communes (3 clusters): {len(data_3['clusters'])}")
    print(f"  [INFO] Nombre de communes (5 clusters): {len(data_5['clusters'])}")

if __name__ == "__main__":
    print("=" * 80)
    print("  GÉNÉRATION DU FICHIER cah_data.js COMPLET")
    print("=" * 80)
    generer_cah_data_js()
    print("=" * 80)
