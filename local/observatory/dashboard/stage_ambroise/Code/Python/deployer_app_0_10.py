"""
Script de déploiement final pour l'application web avec normalisation 0-10
- Copie tous les fichiers nécessaires vers le dossier WEB
- Génère un fichier de configuration
- Crée un rapport récapitulatif
"""

import json
import shutil
from pathlib import Path
import pandas as pd


def main():
    print("=" * 80)
    print("DEPLOIEMENT DE L'APPLICATION WEB (NORMALISATION 0-10)")
    print("=" * 80)
    print()

    web_dir = Path("../WEB")

    if not web_dir.exists():
        print(f"Création du dossier {web_dir}...")
        web_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # ETAPE 1: COPIER LES FICHIERS DE DONNEES
    # ========================================================================
    print("ETAPE 1: Copie des fichiers de données")
    print("-" * 80)

    fichiers_a_copier = {
        'data_scores_0_10.json': 'Données des scores normalisés 0-10',
        'seuils_jenks_0_10.json': 'Seuils Jenks pour les cartes OppChoVec',
        'lisa_oppchovec_0_10_1pct.json': 'LISA OppChoVec 1%',
        'lisa_oppchovec_0_10_5pct.json': 'LISA OppChoVec 5%',
    }

    fichiers_copies = []
    for fichier, description in fichiers_a_copier.items():
        src = Path(fichier)
        if src.exists():
            shutil.copy(src, web_dir / fichier)
            print(f"  [OK] {fichier:45s} - {description}")
            fichiers_copies.append(fichier)
        else:
            print(f"  [ERREUR] {fichier:45s} - Fichier non trouvé")

    # ========================================================================
    # ETAPE 2: GENERER LE FICHIER DE CONFIGURATION
    # ========================================================================
    print("\nETAPE 2: Génération du fichier de configuration")
    print("-" * 80)

    config = {
        'version': '1.0.0',
        'normalisation': {
            'type': '0-10',
            'min': 0,
            'max': 10,
            'description': 'Normalisation linéaire entre 0 et 10'
        },
        'fichiers': {
            'scores': 'data_scores_0_10.json',
            'jenks': 'seuils_jenks_0_10.json',
            'lisa': {
                'oppchovec': {
                    '1pct': 'lisa_oppchovec_0_10_1pct.json',
                    '5pct': 'lisa_oppchovec_0_10_5pct.json'
                }
            }
        },
        'indicateurs': {
            'principal': 'OppChoVec',
            'dimensions': ['Opp', 'Cho', 'Vec']
        },
        'colonnes': {
            'oppchovec': 'OppChoVec_0_10',
            'opp': 'Score_Opp_0_10',
            'cho': 'Score_Cho_0_10',
            'vec': 'Score_Vec_0_10'
        }
    }

    config_file = web_dir / 'config_0_10.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config_file.name}")

    # ========================================================================
    # ETAPE 3: CHARGER LES STATISTIQUES
    # ========================================================================
    print("\nETAPE 3: Chargement des statistiques")
    print("-" * 80)

    # Charger les seuils Jenks
    with open('seuils_jenks_0_10.json', 'r', encoding='utf-8') as f:
        jenks = json.load(f)

    # Charger les données
    with open('data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data_scores = json.load(f)

    print(f"  Nombre de communes: {len(data_scores)}")
    print(f"  Variable analysée: OppChoVec_0_10")

    # ========================================================================
    # ETAPE 4: GENERER LE RAPPORT RECAPITULATIF
    # ========================================================================
    print("\nETAPE 4: Génération du rapport récapitulatif")
    print("-" * 80)

    rapport = []
    rapport.append("=" * 80)
    rapport.append("RAPPORT DE DEPLOIEMENT - APPLICATION WEB NORMALISATION 0-10")
    rapport.append("=" * 80)
    rapport.append("")
    rapport.append(f"Nombre de communes: {len(data_scores)}")
    rapport.append("")
    rapport.append("FICHIERS DEPLOYES:")
    rapport.append("-" * 80)
    for fichier in fichiers_copies:
        rapport.append(f"  [OK] {fichier}")
    rapport.append("")
    rapport.append("SEUILS JENKS (5 classes) - OppChoVec:")
    rapport.append("-" * 80)
    seuils = jenks['OppChoVec_0_10']
    for i, seuil in enumerate(seuils, 1):
        rapport.append(f"  Classe {i}: {seuil:.2f}")
    rapport.append("")

    # Calculer quelques statistiques
    rapport.append("STATISTIQUES OppChoVec (0-10):")
    rapport.append("-" * 80)

    df = pd.read_csv('data_pour_lisa_0_10.csv')
    rapport.append(f"  Min:     {df['OppChoVec_0_10'].min():.2f}")
    rapport.append(f"  Max:     {df['OppChoVec_0_10'].max():.2f}")
    rapport.append(f"  Moyenne: {df['OppChoVec_0_10'].mean():.2f}")
    rapport.append(f"  Mediane: {df['OppChoVec_0_10'].median():.2f}")
    rapport.append(f"  Ecart-type: {df['OppChoVec_0_10'].std():.2f}")
    rapport.append("")

    # Charger les résultats LISA
    rapport.append("RESULTATS LISA (OppChoVec uniquement):")
    rapport.append("-" * 80)

    lisa_files = {
        'LISA 1%': 'lisa_oppchovec_0_10_1pct.json',
        'LISA 5%': 'lisa_oppchovec_0_10_5pct.json',
    }

    for nom, fichier in lisa_files.items():
        if Path(fichier).exists():
            with open(fichier, 'r', encoding='utf-8') as f:
                lisa_data = json.load(f)
            sig_count = sum(1 for v in lisa_data.values() if v['significatif'])
            rapport.append(f"  {nom}: {sig_count}/{len(lisa_data)} communes significatives ({sig_count/len(lisa_data)*100:.1f}%)")

    rapport.append("")
    rapport.append("TOP 10 COMMUNES (OppChoVec 0-10):")
    rapport.append("-" * 80)

    df_sorted = df.sort_values('OppChoVec_0_10', ascending=False).head(10)
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        rapport.append(f"  {i:2d}. {row['Zone']:30s} - {row['OppChoVec_0_10']:.2f}/10")

    rapport.append("")
    rapport.append("=" * 80)
    rapport.append("DEPLOIEMENT TERMINE AVEC SUCCES")
    rapport.append("=" * 80)

    # Sauvegarder le rapport
    rapport_file = 'rapport_deploiement_0_10.txt'
    with open(rapport_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rapport))

    # Copier le rapport vers WEB
    shutil.copy(rapport_file, web_dir / rapport_file)

    # Afficher le rapport
    print('\n'.join(rapport))

    print(f"\n[OK] Rapport sauvegarde: {rapport_file}")
    print(f"[OK] Tous les fichiers copies vers: {web_dir.absolute()}")

    print("\n" + "=" * 80)
    print("DEPLOIEMENT TERMINE")
    print("=" * 80)
    print("\nFichiers disponibles dans ../WEB/:")
    for fichier in sorted(web_dir.glob('*_0_10*')):
        print(f"  - {fichier.name}")


if __name__ == "__main__":
    main()
