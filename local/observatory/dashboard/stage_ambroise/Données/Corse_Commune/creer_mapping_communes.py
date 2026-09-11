"""
Script pour créer le fichier de mapping code INSEE → nom de commune
Utilise l'API geo.data.gouv.fr pour récupérer les données officielles
"""

import requests
import pandas as pd

def recuperer_communes_corse():
    """Récupère les communes de Corse depuis l'API geo.data.gouv.fr"""
    print("Récupération des communes de Corse depuis l'API...")

    departements = ['2A', '2B']
    communes_list = []

    for dept in departements:
        print(f"  Téléchargement des communes du département {dept}...")
        url = f"https://geo.api.gouv.fr/departements/{dept}/communes?format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"    {len(data)} communes trouvées")
            communes_list.extend(data)
        else:
            print(f"    Erreur: {response.status_code}")

    return communes_list

def creer_mapping(communes_list):
    """Crée un DataFrame de mapping code INSEE → nom"""
    mapping_data = []

    for commune in communes_list:
        mapping_data.append({
            'code_commune': commune['code'],
            'nom_commune': commune['nom']
        })

    df = pd.DataFrame(mapping_data)
    df = df.sort_values('code_commune')

    return df

def main():
    print("="*70)
    print("CREATION DU FICHIER DE MAPPING CODE INSEE vers NOM COMMUNE")
    print("="*70)

    # 1. Recuperer les communes
    communes = recuperer_communes_corse()
    print(f"\nTotal: {len(communes)} communes recuperees")

    # 2. Creer le mapping
    df_mapping = creer_mapping(communes)

    # 3. Afficher un apercu
    print("\nApercu du mapping:")
    print(df_mapping.head(10))

    # 4. Sauvegarder
    output_file = "mapping_communes.csv"
    df_mapping.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n[OK] Fichier sauvegarde: {output_file}")
    print(f"  {len(df_mapping)} correspondances creees")

    print("\n" + "="*70)
    print("TERMINE")
    print("="*70)

if __name__ == "__main__":
    main()
