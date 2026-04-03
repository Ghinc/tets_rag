"""
Ground truth pour l'évaluation du pipeline RAG.

Contient les valeurs de référence pour les questions Factual Accuracy et Binary,
ainsi que les groupes de questions pour la robustesse sémantique.
"""

import sys

# === GROUND TRUTH pour Factual Accuracy ===
# Format: question (str normalisée) → valeur attendue (float ou str)
# - float → comparaison numérique (MAE)
# - str  → vérification sémantique via LLM
GROUND_TRUTH = {
    # OppChoVec scores — Ajaccio
    "Quel est le score moyen de bien-être à Ajaccio ?": 5.78,
    "Quelle dimension obtient la note la plus faible ?": "Choix (Cho) : 1.27",
    "Quel est le score OppChoVec d'Ajaccio, par catégorie ?": "Opp=10.00, Cho=1.27, Vec=4.80, Total=5.78",
    # OppChoVec scores — comparaisons
    "Le score de bien-être à Lozzi est-il supérieur ou inférieur à la moyenne des communes du Niolu ?": "inférieur",  # 2.35 vs 2.67
    "De combien le score moyen de bien-être à Ajaccio diffère-t-il de celui de Bastia ?": 3.06,  # 5.78 - 2.72
    "Guargualé est-elle au-dessus ou en dessous de la moyenne régionale, et de combien ?": "en dessous, écart de 1.34",
    # CAPA — valeurs calculées depuis ChromaDB (complétées par compute_capa_ground_truth)
    "Quelle est la commune de la Communauté d'Agglomération du Pays Ajaccien (CAPA) avec le meilleur score OppChoVec global ?": "Afa",
    "Quelle est la commune de la CAPA avec le meilleur score Opportunités (Opp) ?": "Ajaccio",
    "Quelle est la commune de la CAPA avec le meilleur score Choix (Cho) ?": "Cuttoli-Corticchiato",
    "Quelle est la commune de la CAPA avec le meilleur score Vécu (Vec) ?": "Afa",
}

# === GROUND TRUTH pour Binary / 0-1 ===
# Format: question (str) → label attendu (str)
BINARY_EXPECTED = {
    "Quelle commune obtient le score objectif global le plus élevé entre Ajaccio et Bastia ?": "Ajaccio",
    "Le score de bien-être à Lozzi est-il supérieur ou inférieur à la moyenne des communes du Niolu ?": "inférieur",
}

# === GROUPES Robustesse Sémantique ===
# Format: nom_groupe → liste de numéros de ligne Excel (1-indexed, inclut la ligne header)
ROBUSTNESS_GROUPS = {
    "reformulations_paraphrastiques": [57, 58, 59],
    "variations_syntaxiques": [60, 61, 62],
    "reformulations_indirectes": [63, 64, 65],
    "ambiguite_et_bruit_lexical": [66, 67, 68],
}

# Contexte sur les questions sans ground truth connu (pour le rapport)
GT_COMMENTS = {
    "Combien d'habitants ont répondu à l'enquête à Ajaccio ?":
        "Ground truth nécessite de compter dans le CSV de l'enquête — non vérifiable automatiquement.",
    "De combien de services de proximité dispose la ville d'Ajaccio":
        "Ground truth issu des données équipements — vérification manuelle requise.",
    "Quel est l'écart moyen entre les communes rurales et urbaines":
        "Pas de classification rural/urbain explicite dans les données — résultat estimatif.",
    "Quel est l'écart de score environnement entre Ajaccio et la":
        "Dimension 'environnement' est qualitative dans l'enquête, pas dans OppChoVec — MAE difficile.",
}


def compute_capa_ground_truth(chroma_path: str = "./chroma_portrait") -> dict:
    """
    Calcule automatiquement les communes CAPA avec le meilleur score par dimension.
    Retourne un dict complémentaire à GROUND_TRUTH (override si différent).
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)

        # Récupérer la liste des communes CAPA depuis communes_geo
        geo = client.get_collection("communes_geo")
        geo_res = geo.get(include=["metadatas"])
        capa_communes = {
            m["commune"] for m in geo_res["metadatas"]
            if "Ajaccien" in str(m.get("epci", ""))
        }
        if not capa_communes:
            print("[AVERTISSEMENT] Aucune commune CAPA trouvée dans communes_geo", file=sys.stderr)
            return {}

        # Récupérer les scores OppChoVec pour ces communes
        opp = client.get_collection("oppchovec_scores")
        opp_res = opp.get(where={"source": "oppchovec_betti_0_10"}, include=["metadatas"])
        capa_scores = [m for m in opp_res["metadatas"] if m["commune"] in capa_communes]
        if not capa_scores:
            return {}

        best_global = max(capa_scores, key=lambda x: x["oppchovec_0_10"])["commune"]
        best_opp = max(capa_scores, key=lambda x: x["opp_0_10"])["commune"]
        best_cho = max(capa_scores, key=lambda x: x["cho_0_10"])["commune"]
        best_vec = max(capa_scores, key=lambda x: x["vec_0_10"])["commune"]

        return {
            "Quelle est la commune de la Communauté d'Agglomération du Pays Ajaccien (CAPA) avec le meilleur score OppChoVec global ?": best_global,
            "Quelle est la commune de la CAPA avec le meilleur score Opportunités (Opp) ?": best_opp,
            "Quelle est la commune de la CAPA avec le meilleur score Choix (Cho) ?": best_cho,
            "Quelle est la commune de la CAPA avec le meilleur score Vécu (Vec) ?": best_vec,
        }

    except Exception as e:
        print(f"[AVERTISSEMENT] Calcul ground truth CAPA échoué : {e}", file=sys.stderr)
        return {}


def get_full_ground_truth(chroma_path: str = "./chroma_portrait") -> dict:
    """Retourne le ground truth complet (statique + CAPA calculé depuis ChromaDB)."""
    gt = dict(GROUND_TRUTH)
    gt.update(compute_capa_ground_truth(chroma_path))
    return gt


def find_ground_truth(question: str, gt: dict) -> tuple:
    """
    Cherche la valeur ground truth pour une question.
    Utilise une correspondance exacte puis une correspondance par sous-chaîne.
    Retourne (valeur, trouvé: bool).
    """
    q = question.strip()
    if q in gt:
        return gt[q], True
    # Correspondance partielle (question tronquée ou légèrement différente)
    q_lower = q.lower()
    for key, val in gt.items():
        if q_lower in key.lower() or key.lower() in q_lower:
            return val, True
    return None, False


if __name__ == "__main__":
    print("=== GROUND TRUTH COMPLET ===")
    gt = get_full_ground_truth()
    for q, v in gt.items():
        print(f"  {q[:70]:72s} -> {v!r}")
    print(f"\n  Total : {len(gt)} entrees")

    print("\n=== BINARY EXPECTED ===")
    for q, v in BINARY_EXPECTED.items():
        print(f"  {q[:70]:72s} -> {v!r}")

    print("\n=== GROUPES ROBUSTESSE SÉMANTIQUE ===")
    for g, rows in ROBUSTNESS_GROUPS.items():
        print(f"  {g}: lignes Excel {rows}")
