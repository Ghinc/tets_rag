"""
Script pour harmoniser les noms de communes entre le JSON et le GeoJSON
Corrections nécessaires :
- "La Porta" -> "Porta" (pour correspondre au GeoJSON)
- "L'Île-Rousse" -> "Île-Rousse" (pour correspondre au GeoJSON)
"""

import json
import shutil
from datetime import datetime

def harmoniser_noms():
    """Harmonise les noms de communes dans le fichier JSON"""

    json_path = '../WEB/data_scores_0_10.json'

    # Créer une sauvegarde
    backup_path = f'../WEB/data_scores_0_10_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    shutil.copy(json_path, backup_path)
    print(f"[INFO] Sauvegarde créée: {backup_path}")

    # Charger les données
    print(f"\n[INFO] Chargement du fichier JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  [OK] {len(data)} communes chargées")

    # Dictionnaire de correspondance
    mapping = {
        "La Porta": "Porta",
        "L'Île-Rousse": "Île-Rousse"
    }

    # Appliquer les modifications
    print(f"\n[INFO] Application des modifications...")
    for old_name, new_name in mapping.items():
        if old_name in data:
            data[new_name] = data.pop(old_name)
            print(f"  [OK] '{old_name}' -> '{new_name}'")
        else:
            print(f"  [WARN] '{old_name}' non trouvé dans les données")

    # Sauvegarder le fichier modifié
    print(f"\n[INFO] Sauvegarde du fichier modifié...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Fichier sauvegardé: {json_path}")
    print(f"\n[OK] Harmonisation terminée! {len(data)} communes dans le fichier final")

    return data

if __name__ == "__main__":
    print("=" * 80)
    print("  HARMONISATION DES NOMS DE COMMUNES")
    print("=" * 80)

    harmoniser_noms()

    print("\n" + "=" * 80)
    print("[OK] TRAITEMENT TERMINE")
    print("=" * 80)
