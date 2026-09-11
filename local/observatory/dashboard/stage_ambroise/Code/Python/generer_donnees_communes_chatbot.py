"""
Génère les données pour le chatbot DumèGPT :
- Un fichier JSON complet avec toutes les données par commune
- Un dossier avec un fichier .txt par commune résumant toutes les informations
"""

import json
import os
from pathlib import Path

def charger_donnees():
    """Charge les données depuis data_scores_0_10.json"""
    print("[INFO] Chargement des données...")

    with open('../WEB/data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  [OK] {len(data)} communes chargées")
    return data

def generer_json_complet(data):
    """Génère le JSON complet avec toutes les données pour chaque commune"""
    print("\n[INFO] Génération du JSON complet...")

    # Structurer les données de manière claire
    donnees_completes = {}

    for commune, scores in data.items():
        donnees_completes[commune] = {
            # Scores globaux (0-10)
            "scores_globaux": {
                "OppChoVec": round(scores.get('OppChoVec_0_10', 0), 2),
                "Opportunités": round(scores.get('Score_Opp_0_10', 0), 2),
                "Choix": round(scores.get('Score_Cho_0_10', 0), 2),
                "Vécu": round(scores.get('Score_Vec_0_10', 0), 2)
            },

            # Indicateurs Opportunités détaillés
            "opportunites": {
                "Opp1_Education": round(scores.get('Opp1', 0), 2),
                "Opp2_Inegalites": round(scores.get('Opp2', 0), 4),
                "Opp3_Mobilite": round(scores.get('Opp3', 0), 2),
                "Opp4_Connectivite": round(scores.get('Opp4', 0), 2)
            },

            # Indicateurs Choix détaillés
            "choix": {
                "Cho1_QuartierPrioritaire": round(scores.get('Cho1', 0), 2),
                "Cho2_ParticipationElectorale": round(scores.get('Cho2', 0), 2)
            },

            # Indicateurs Vécu détaillés
            "vecu": {
                "Vec1_NiveauVie": int(scores.get('Vec1', 0)),
                "Vec2_Logement": round(scores.get('Vec2', 0), 4),
                "Vec3_StatutSocial": round(scores.get('Vec3', 0), 4),
                "Vec4_Emploi": int(scores.get('Vec4', 0))
            }
        }

    # Sauvegarder le JSON
    output_path = '../WEB/donnees_completes_communes.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(donnees_completes, f, indent=2, ensure_ascii=False)

    print(f"  [OK] JSON complet sauvegardé: {output_path}")
    return donnees_completes

def generer_fichiers_txt(donnees_completes):
    """Génère un fichier .txt par commune avec un résumé des informations"""
    print("\n[INFO] Génération des fichiers texte par commune...")

    # Créer le dossier de sortie
    output_dir = Path('../OUTPUT/communes_chatbot')
    output_dir.mkdir(parents=True, exist_ok=True)

    for commune, data in donnees_completes.items():
        # Créer le contenu du fichier
        contenu = f"""{'='*80}
FICHE COMMUNE : {commune.upper()}
{'='*80}

📊 SCORES GLOBAUX (échelle 0-10)
{'─'*80}
• OppChoVec (Score global)    : {data['scores_globaux']['OppChoVec']}/10
• Opportunités                : {data['scores_globaux']['Opportunités']}/10
• Choix                       : {data['scores_globaux']['Choix']}/10
• Vécu                        : {data['scores_globaux']['Vécu']}/10


🎯 OPPORTUNITÉS - Détail des indicateurs
{'─'*80}
Score global Opportunités : {data['scores_globaux']['Opportunités']}/10

Composantes :
  Opp1 - Niveau d'éducation          : {data['opportunites']['Opp1_Education']}
           → Mesure le niveau moyen d'éducation de la population

  Opp2 - Inégalités (Indice de Theil) : {data['opportunites']['Opp2_Inegalites']}
           → Plus le score est bas, moins il y a d'inégalités

  Opp3 - Mobilité                     : {data['opportunites']['Opp3_Mobilite']}
           → Accès aux transports et taux d'équipement automobile

  Opp4 - Connectivité numérique       : {data['opportunites']['Opp4_Connectivite']}
           → Couverture 4G et accès au très haut débit


🗳️  CHOIX - Détail des indicateurs
{'─'*80}
Score global Choix : {data['scores_globaux']['Choix']}/10

Composantes :
  Cho1 - Quartier prioritaire         : {data['choix']['Cho1_QuartierPrioritaire']}
           → Présence de quartiers prioritaires de la politique de la ville

  Cho2 - Participation électorale     : {data['choix']['Cho2_ParticipationElectorale']}
           → Taux de participation aux élections


🏠 VÉCU - Détail des indicateurs
{'─'*80}
Score global Vécu : {data['scores_globaux']['Vécu']}/10

Composantes :
  Vec1 - Niveau de vie médian         : {data['vecu']['Vec1_NiveauVie']} €/an
           → Revenu médian de la commune

  Vec2 - Qualité du logement          : {data['vecu']['Vec2_Logement']}
           → Nombre de pièces, décence, type de logement

  Vec3 - Statut social (CSP)          : {data['vecu']['Vec3_StatutSocial']}
           → Répartition des catégories socio-professionnelles

  Vec4 - Emploi                       : {data['vecu']['Vec4_Emploi']} établissements
           → Nombre d'établissements employeurs


{'='*80}
INTERPRÉTATION DES SCORES
{'='*80}

Score OppChoVec = {data['scores_globaux']['OppChoVec']}/10
"""

        # Interprétation du score global
        score_global = data['scores_globaux']['OppChoVec']
        if score_global >= 7:
            interpretation = "🟢 Excellente qualité de vie globale"
        elif score_global >= 5:
            interpretation = "🟡 Bonne qualité de vie avec quelques axes d'amélioration"
        elif score_global >= 3:
            interpretation = "🟠 Qualité de vie moyenne, plusieurs axes nécessitent attention"
        else:
            interpretation = "🔴 Difficultés importantes sur plusieurs dimensions"

        contenu += f"{interpretation}\n\n"

        # Analyse des forces et faiblesses
        scores = {
            'Opportunités': data['scores_globaux']['Opportunités'],
            'Choix': data['scores_globaux']['Choix'],
            'Vécu': data['scores_globaux']['Vécu']
        }

        meilleur = max(scores.items(), key=lambda x: x[1])
        moins_bon = min(scores.items(), key=lambda x: x[1])

        contenu += f"Points forts :\n  • {meilleur[0]} : {meilleur[1]}/10\n\n"
        contenu += f"Axes d'amélioration :\n  • {moins_bon[0]} : {moins_bon[1]}/10\n\n"

        contenu += f"{'='*80}\n"
        contenu += f"Document généré automatiquement pour le chatbot DumèGPT\n"
        contenu += f"{'='*80}\n"

        # Sauvegarder le fichier
        filename = output_dir / f"{commune.replace(' ', '_')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(contenu)

    print(f"  [OK] {len(donnees_completes)} fichiers texte générés dans {output_dir}")
    return output_dir

def afficher_statistiques(donnees_completes):
    """Affiche quelques statistiques sur les données générées"""
    print("\n[INFO] Statistiques des données :")

    scores_opp = [d['scores_globaux']['Opportunités'] for d in donnees_completes.values()]
    scores_cho = [d['scores_globaux']['Choix'] for d in donnees_completes.values()]
    scores_vec = [d['scores_globaux']['Vécu'] for d in donnees_completes.values()]
    scores_global = [d['scores_globaux']['OppChoVec'] for d in donnees_completes.values()]

    print(f"\n  Opportunités :")
    print(f"    Moyenne : {sum(scores_opp)/len(scores_opp):.2f}/10")
    print(f"    Min     : {min(scores_opp):.2f}/10")
    print(f"    Max     : {max(scores_opp):.2f}/10")

    print(f"\n  Choix :")
    print(f"    Moyenne : {sum(scores_cho)/len(scores_cho):.2f}/10")
    print(f"    Min     : {min(scores_cho):.2f}/10")
    print(f"    Max     : {max(scores_cho):.2f}/10")

    print(f"\n  Vécu :")
    print(f"    Moyenne : {sum(scores_vec)/len(scores_vec):.2f}/10")
    print(f"    Min     : {min(scores_vec):.2f}/10")
    print(f"    Max     : {max(scores_vec):.2f}/10")

    print(f"\n  OppChoVec (global) :")
    print(f"    Moyenne : {sum(scores_global)/len(scores_global):.2f}/10")
    print(f"    Min     : {min(scores_global):.2f}/10")
    print(f"    Max     : {max(scores_global):.2f}/10")

def main():
    print("=" * 80)
    print("  GÉNÉRATION DES DONNÉES POUR LE CHATBOT DUMÈGPT")
    print("=" * 80)

    # Charger les données
    data = charger_donnees()

    # Générer le JSON complet
    donnees_completes = generer_json_complet(data)

    # Générer les fichiers .txt par commune
    output_dir = generer_fichiers_txt(donnees_completes)

    # Afficher les statistiques
    afficher_statistiques(donnees_completes)

    print("\n" + "=" * 80)
    print("[OK] GÉNÉRATION TERMINÉE")
    print(f"     - JSON complet : ../WEB/donnees_completes_communes.json")
    print(f"     - Fichiers .txt : {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
