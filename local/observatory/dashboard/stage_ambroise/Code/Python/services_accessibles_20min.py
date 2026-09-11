"""
Calcule le nombre de services accessibles à moins de 20 minutes en voiture
pour chaque commune de Corse.

Utilise la Base Permanente des Équipements (BPE) de l'INSEE pour avoir
TOUS les services, y compris les petits commerces.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path
import zipfile
import io

# Configuration
DATA_DIR = Path("../../Données/Corse_Commune")
MAPPING_FILE = DATA_DIR / "mapping_communes.csv"
COORDS_FILE = "communes_corse_coordonnees.csv"

# URL de la BPE INSEE (version 2024 - la plus récente)
BPE_URL = "https://www.insee.fr/fr/statistiques/fichier/8217525/BPE24.zip"

# API OSRM pour calcul des temps de trajet
OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"

# Seuil de temps (minutes)
SEUIL_TEMPS = 20


def telecharger_bpe() -> pd.DataFrame:
    """
    Télécharge et charge la Base Permanente des Équipements de l'INSEE
    Filtre pour garder uniquement la Corse (départements 2A et 2B)

    Returns:
        DataFrame avec tous les équipements de Corse
    """
    print("Téléchargement de la Base Permanente des Équipements (BPE)...")
    print("(Fichier ~100 Mo, cela peut prendre quelques minutes)")

    try:
        # Télécharger le fichier ZIP
        response = requests.get(BPE_URL, timeout=300)
        response.raise_for_status()

        # Extraire le CSV du ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Lister tous les fichiers et chercher le CSV principal
            all_files = z.namelist()
            print(f"  Fichiers dans le ZIP: {all_files}")

            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__')]

            if not csv_files:
                print("Erreur: Aucun fichier CSV trouvé dans le ZIP")
                return pd.DataFrame()

            # Prendre le plus gros fichier CSV (le fichier principal)
            csv_file = max(csv_files, key=lambda f: z.getinfo(f).file_size)
            print(f"  Extraction de {csv_file}...")

            with z.open(csv_file) as f:
                # Charger le CSV avec le bon encodage
                df = pd.read_csv(f, sep=';', encoding='utf-8', low_memory=False)

        print(f"  {len(df)} équipements chargés (France entière)")

        # Afficher les colonnes disponibles
        print(f"  Colonnes: {list(df.columns)[:10]}...")

        # Filtrer pour la Corse uniquement (départements 2A et 2B)
        # Chercher la colonne département
        dep_col = None
        for col in ['DEP', 'DEPCOM', 'dep', 'DESCRIPT']:
            if col in df.columns:
                dep_col = col
                break

        if dep_col is None:
            print("Erreur: Colonne département non trouvée")
            return pd.DataFrame()

        # Filtrer selon le type de colonne
        if dep_col == 'DEPCOM':
            # Si c'est DEPCOM, prendre les codes commençant par 2A ou 2B
            df_corse = df[df[dep_col].astype(str).str.startswith(('2A', '2B'))].copy()
        else:
            # Sinon, filtrer directement
            df_corse = df[df[dep_col].isin(['2A', '2B'])].copy()

        print(f"  {len(df_corse)} équipements en Corse")

        # Sauvegarder localement pour réutilisation
        df_corse.to_csv('bpe_corse.csv', index=False, encoding='utf-8')
        print("  [OK] BPE Corse sauvegardée: bpe_corse.csv")

        return df_corse

    except Exception as e:
        print(f"Erreur lors du téléchargement de la BPE: {e}")
        print("\nVous pouvez télécharger manuellement depuis:")
        print("https://www.insee.fr/fr/statistiques/3568638")
        return pd.DataFrame()


def charger_bpe_corse() -> pd.DataFrame:
    """
    Charge la BPE Corse (depuis fichier local ou téléchargement)

    Returns:
        DataFrame avec les équipements de Corse
    """
    # Essayer de charger depuis fichier local
    if Path('bpe_corse.csv').exists():
        print("Chargement de la BPE depuis le fichier local...")
        df = pd.read_csv('bpe_corse.csv', encoding='utf-8')
        print(f"  {len(df)} équipements chargés")
        return df
    else:
        # Sinon télécharger
        return telecharger_bpe()


def calculer_matrice_distances_osrm(coords_sources: List[Tuple[float, float]],
                                   coords_destinations: List[Tuple[float, float]],
                                   max_points: int = 100) -> np.ndarray:
    """
    Calcule la matrice des temps de trajet entre sources et destinations via OSRM

    Args:
        coords_sources: Liste de tuples (longitude, latitude) des communes
        coords_destinations: Liste de tuples (longitude, latitude) des services
        max_points: Nombre maximum de points par requête OSRM

    Returns:
        Matrice numpy des temps de trajet en minutes
    """
    n_sources = len(coords_sources)
    n_dest = len(coords_destinations)

    matrice = np.full((n_sources, n_dest), np.nan)

    print(f"  Calcul de la matrice {n_sources} communes x {n_dest} services...")

    # Traiter par batch pour ne pas dépasser les limites de l'API
    batch_size = 25  # Batch réduit pour éviter les erreurs

    for i in range(0, n_sources, batch_size):
        end_i = min(i + batch_size, n_sources)
        batch_sources = coords_sources[i:end_i]

        for j in range(0, n_dest, batch_size):
            end_j = min(j + batch_size, n_dest)
            batch_dest = coords_destinations[j:end_j]

            # Construire la requête
            all_coords = batch_sources + batch_dest
            coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])

            sources_idx = ";".join([str(k) for k in range(len(batch_sources))])
            dest_idx = ";".join([str(k) for k in range(len(batch_sources), len(all_coords))])

            url = f"{OSRM_URL}{coords_str}?sources={sources_idx}&destinations={dest_idx}"

            try:
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if data.get('code') == 'Ok':
                        durations = data.get('durations', [])

                        # Remplir la matrice (convertir secondes en minutes)
                        for idx_s, row in enumerate(durations):
                            for idx_d, duration in enumerate(row):
                                if duration is not None:
                                    matrice[i + idx_s, j + idx_d] = duration / 60.0

                time.sleep(1.2)  # Pause pour respecter les limites de l'API

                # Afficher la progression
                progress = ((i * n_dest + end_j) / (n_sources * n_dest)) * 100
                if progress % 10 < 1:
                    print(f"    Progression: {progress:.1f}%")

            except Exception as e:
                print(f"    Erreur pour batch ({i}-{end_i}, {j}-{end_j}): {e}")
                continue

    return matrice


def compter_services_accessibles(df_communes: pd.DataFrame,
                                 df_equipements: pd.DataFrame,
                                 seuil_minutes: int = 20) -> pd.DataFrame:
    """
    Compte le nombre de services accessibles à moins de X minutes pour chaque commune

    Args:
        df_communes: DataFrame des communes avec coordonnées
        df_equipements: DataFrame BPE avec les équipements
        seuil_minutes: Seuil de temps en minutes

    Returns:
        DataFrame avec le nombre de services accessibles par commune
    """
    print(f"\nCalcul des services accessibles à moins de {seuil_minutes} minutes...")

    # Préparer les coordonnées
    coords_communes = [(row['longitude'], row['latitude'])
                      for _, row in df_communes.iterrows()]

    # Pour les équipements, vérifier si les coordonnées WGS84 sont déjà présentes
    if 'LONGITUDE' in df_equipements.columns and 'LATITUDE' in df_equipements.columns:
        print("  Coordonnées WGS84 directement disponibles (LONGITUDE, LATITUDE)")
        # Renommer pour uniformiser
        df_equipements = df_equipements.rename(columns={'LONGITUDE': 'longitude', 'LATITUDE': 'latitude'})
    elif 'longitude' not in df_equipements.columns or 'latitude' not in df_equipements.columns:
        # Sinon, chercher les coordonnées Lambert et convertir
        coord_x_col = None
        coord_y_col = None

        for col_x in ['LAMBERT_X', 'X', 'COORDONNEES_X', 'lambert_x', 'x']:
            if col_x in df_equipements.columns:
                coord_x_col = col_x
                break

        for col_y in ['LAMBERT_Y', 'Y', 'COORDONNEES_Y', 'lambert_y', 'y']:
            if col_y in df_equipements.columns:
                coord_y_col = col_y
                break

        if coord_x_col and coord_y_col:
            print(f"  Coordonnées Lambert trouvées: {coord_x_col}, {coord_y_col}")
            # Renommer pour uniformiser
            df_equipements = df_equipements.rename(columns={coord_x_col: 'LAMBERT_X', coord_y_col: 'LAMBERT_Y'})
            # Convertir Lambert 93 vers WGS84 (lat/lon)
            df_equipements = convertir_lambert_vers_wgs84(df_equipements)
        else:
            print("  Attention: Colonnes de coordonnées non trouvées")
            df_equipements['longitude'] = np.nan
            df_equipements['latitude'] = np.nan

    # Filtrer les équipements avec coordonnées valides
    df_equip_valides = df_equipements.dropna(subset=['longitude', 'latitude']).copy()

    print(f"  {len(df_equip_valides)} équipements avec coordonnées valides")

    if len(df_equip_valides) == 0:
        print("  Erreur: Aucun équipement avec coordonnées valides!")
        print("  Impossible de calculer les temps d'accès.")
        return pd.DataFrame([{
            'code_commune': row['code_commune'],
            'nom_commune': row['nom_commune'],
            'nb_services_20min': 0,
            'temps_service_plus_proche_min': np.nan,
            'temps_moyen_services_min': np.nan
        } for _, row in df_communes.iterrows()])

    coords_equipements = [(row['longitude'], row['latitude'])
                         for _, row in df_equip_valides.iterrows()]

    # Calculer la matrice de distances
    matrice = calculer_matrice_distances_osrm(coords_communes, coords_equipements)

    # Compter les services accessibles
    resultats = []

    for idx, row in df_communes.iterrows():
        temps_vers_services = matrice[idx, :]

        # Compter combien de services sont à moins de X minutes
        services_accessibles = np.sum(temps_vers_services <= seuil_minutes)

        # Statistiques sur les temps d'accès
        if len(temps_vers_services) > 0 and not np.all(np.isnan(temps_vers_services)):
            temps_min = np.nanmin(temps_vers_services)
            temps_moyen = np.nanmean(temps_vers_services)
        else:
            temps_min = np.nan
            temps_moyen = np.nan

        resultats.append({
            'code_commune': row['code_commune'],
            'nom_commune': row['nom_commune'],
            'nb_services_20min': services_accessibles,
            'temps_service_plus_proche_min': temps_min,
            'temps_moyen_services_min': temps_moyen
        })

        if (idx + 1) % 50 == 0:
            print(f"    Traité: {idx + 1}/{len(df_communes)} communes")

    return pd.DataFrame(resultats)


def convertir_lambert_vers_wgs84(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les coordonnées Lambert 93 en WGS84 (latitude/longitude)

    Utilise pyproj pour la conversion

    Args:
        df: DataFrame avec colonnes LAMBERT_X et LAMBERT_Y

    Returns:
        DataFrame avec colonnes longitude et latitude ajoutées
    """
    print("  Conversion Lambert 93 -> WGS84...")

    try:
        from pyproj import Transformer

        # Créer le transformateur Lambert 93 (EPSG:2154) -> WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

        # Filtrer les lignes avec coordonnées valides
        mask_valide = df['LAMBERT_X'].notna() & df['LAMBERT_Y'].notna()

        # Initialiser les colonnes
        df['longitude'] = np.nan
        df['latitude'] = np.nan

        if mask_valide.sum() > 0:
            # Convertir uniquement les coordonnées valides
            x_valides = df.loc[mask_valide, 'LAMBERT_X'].values
            y_valides = df.loc[mask_valide, 'LAMBERT_Y'].values

            # Transformation par lot (plus rapide)
            lon_valides, lat_valides = transformer.transform(x_valides, y_valides)

            # Assigner les valeurs converties
            df.loc[mask_valide, 'longitude'] = lon_valides
            df.loc[mask_valide, 'latitude'] = lat_valides

        nb_converties = df['longitude'].notna().sum()
        print(f"    {nb_converties} coordonnées converties")

    except ImportError:
        print("    PyProj non disponible, installation recommandée: pip install pyproj")
        df['longitude'] = np.nan
        df['latitude'] = np.nan
    except Exception as e:
        print(f"    Erreur lors de la conversion: {e}")
        df['longitude'] = np.nan
        df['latitude'] = np.nan

    return df


def calculer_indicateur_accessibilite_services(nb_services: int) -> float:
    """
    Convertit un nombre de services en indicateur normalisé (0-1)

    Args:
        nb_services: Nombre de services accessibles à moins de 20 minutes

    Returns:
        Score entre 0 et 1
    """
    if pd.isna(nb_services):
        return np.nan

    # Normalisation logarithmique
    # 0 services = 0
    # 10 services = ~0.5
    # 100 services = ~0.8
    # 1000+ services = 1

    if nb_services == 0:
        return 0.0
    elif nb_services >= 1000:
        return 1.0
    else:
        return np.log10(nb_services + 1) / 3.0  # log10(1000) = 3


def main():
    """
    Fonction principale
    """
    print("="*80)
    print("SERVICES ACCESSIBLES A MOINS DE 20 MINUTES - CORSE")
    print("="*80)

    # 1. Charger les coordonnées des communes
    if not Path(COORDS_FILE).exists():
        print(f"Erreur: {COORDS_FILE} non trouvé")
        print("Exécutez d'abord temps_acces_services.py")
        return

    df_communes = pd.read_csv(COORDS_FILE)
    print(f"\n{len(df_communes)} communes chargées")

    # 2. Charger la BPE
    df_equipements = charger_bpe_corse()

    if df_equipements.empty:
        print("Erreur: BPE non chargée")
        return

    # Afficher les types d'équipements disponibles
    if 'TYPEQU' in df_equipements.columns:
        types_equip = df_equipements['TYPEQU'].value_counts()
        print(f"\nTypes d'équipements (top 20):")
        for equip, count in types_equip.head(20).items():
            print(f"  {equip}: {count}")

    # 3. Compter les services accessibles
    df_resultats = compter_services_accessibles(df_communes, df_equipements, SEUIL_TEMPS)

    # 4. Calculer l'indicateur d'accessibilité
    print("\nCalcul de l'indicateur d'accessibilité...")
    df_resultats['indicateur_accessibilite_services'] = df_resultats['nb_services_20min'].apply(
        calculer_indicateur_accessibilite_services
    )

    # 5. Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)

    nb_services_valides = df_resultats['nb_services_20min'].dropna()

    if len(nb_services_valides) > 0:
        print(f"\nServices accessibles à moins de {SEUIL_TEMPS} minutes:")
        print(f"  Moyenne: {nb_services_valides.mean():.1f} services")
        print(f"  Médiane: {nb_services_valides.median():.1f} services")
        print(f"  Min: {nb_services_valides.min():.0f} services ({df_resultats.loc[nb_services_valides.idxmin(), 'nom_commune']})")
        print(f"  Max: {nb_services_valides.max():.0f} services ({df_resultats.loc[nb_services_valides.idxmax(), 'nom_commune']})")

        print("\nTop 10 communes avec le plus de services accessibles:")
        top10 = df_resultats.nlargest(10, 'nb_services_20min')
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"  {i:2d}. {row['nom_commune']:30s} - {row['nb_services_20min']:.0f} services")

        print("\nTop 10 communes avec le moins de services accessibles:")
        bottom10 = df_resultats.nsmallest(10, 'nb_services_20min')
        for i, (_, row) in enumerate(bottom10.iterrows(), 1):
            print(f"  {i:2d}. {row['nom_commune']:30s} - {row['nb_services_20min']:.0f} services")

    # 6. Export
    print("\n" + "="*80)
    print("EXPORT")
    print("="*80)

    # CSV
    output_csv = 'services_accessibles_20min.csv'
    df_resultats.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n[OK] Fichier CSV: {output_csv}")

    # Excel
    output_excel = 'services_accessibles_20min.xlsx'
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_resultats.to_excel(writer, sheet_name='Services_20min', index=False)

        # Statistiques
        stats = df_resultats['nb_services_20min'].describe()
        stats.to_excel(writer, sheet_name='Statistiques')

    print(f"[OK] Fichier Excel: {output_excel}")

    # JSON
    output_json = 'services_accessibles_20min.json'
    data_dict = df_resultats.set_index('nom_commune').to_dict(orient='index')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] Fichier JSON: {output_json}")

    print("\n" + "="*80)
    print("TERMINE")
    print("="*80)


if __name__ == "__main__":
    main()
