"""
Indexation des nouvelles sources de données dans chroma_portrait.

Collections créées :
  - portrait_entretiens    : entretiens semi-directifs (Léa) — quali
  - enquete_responses      : réponses individuelles questionnaire — quanti
  - enquete_scores_commune : scores moyens + dimensions par commune — quanti
  - communes_wiki          : résumés Wikipedia des communes corses — mixte

Usage :
    python index_new_sources.py [--collection all|entretiens|enquete|wiki]
"""

import os
import re
import argparse
import pandas as pd
from typing import List, Dict, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

CHROMA_PATH = "./chroma_portrait"
EMBED_MODEL_PATH = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# ============================================================
# Embedding
# ============================================================

def get_embed_model():
    from sentence_transformers import SentenceTransformer
    print("Chargement BGE-M3...")
    return SentenceTransformer(EMBED_MODEL_PATH)


def embed_batch(model, texts: List[str], prefix: str = "passage: ") -> List[List[float]]:
    return [model.encode(prefix + t).tolist() for t in texts]


# ============================================================
# 1. Entretiens semi-directifs (Léa)
# ============================================================

def load_entretiens(filepath: str = "./entretiens_lea.txt") -> Tuple[List[str], List[Dict]]:
    """
    Parse entretiens_lea.txt par entretien, puis découpe chaque entretien
    en chunks de ~600 chars en regroupant des paires Q/R consécutives.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Séparer par entretien
    pattern = r"### Commune : ([^|]+) \| Entretien (\d+)\s*\n(.*?)(?=### Commune|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    print(f"  {len(matches)} entretiens trouvés")

    texts, metas = [], []
    for commune, num, body in matches:
        commune = commune.strip()
        body = body.strip()

        # Découper en blocs Q/R (chaque "Q :" démarre un nouveau bloc)
        qa_blocks = re.split(r"\n(?=Q\s*:)", body)
        qa_blocks = [b.strip() for b in qa_blocks if b.strip()]

        # Regrouper en chunks de ~600 chars
        chunk, chunk_idx = "", 0
        for block in qa_blocks:
            if len(chunk) + len(block) > 600 and chunk:
                texts.append(chunk.strip())
                metas.append({
                    "commune": commune,
                    "nom": commune,
                    "num_entretien": int(num),
                    "chunk_idx": chunk_idx,
                    "source_type": "entretien",
                    "data_type": "quali",
                })
                chunk_idx += 1
                chunk = block + "\n\n"
            else:
                chunk += block + "\n\n"
        if chunk.strip():
            texts.append(chunk.strip())
            metas.append({
                "commune": commune,
                "nom": commune,
                "num_entretien": int(num),
                "chunk_idx": chunk_idx,
                "source_type": "entretien",
                "data_type": "quali",
            })

    print(f"  {len(texts)} chunks d'entretiens générés")
    return texts, metas


def index_entretiens(client, model):
    col = client.get_or_create_collection("portrait_entretiens")
    existing = set(col.get(include=[])["ids"])
    print(f"Collection portrait_entretiens : {col.count()} docs existants")

    texts, metas = load_entretiens()
    ids = [
        f"entretien_{m['commune'].lower().replace(' ', '_')}_{m['num_entretien']}_{m['chunk_idx']}"
        for m in metas
    ]

    to_add = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if id_ not in existing]
    if not to_add:
        print("  Rien à ajouter (déjà indexé)")
        return

    ids_a, texts_a, metas_a = zip(*to_add)
    embeddings = embed_batch(model, list(texts_a))
    col.add(ids=list(ids_a), documents=list(texts_a), embeddings=embeddings, metadatas=list(metas_a))
    print(f"  [OK] {len(ids_a)} chunks ajoutés -> total {col.count()}")


# ============================================================
# 2. Réponses individuelles questionnaire
# ============================================================

# Mapping colonnes score Likert -> label court
SCORE_COLS = {
    "Les services de transports": "Transports",
    "L'accès à l'éducation": "Education",
    "La couverture des réseaux téléphoniques": "Reseaux",
    "Les institutions étatiques (niveau de confiance)": "Institutions",
    "Le tourisme (ressenti localement)": "Tourisme",
    "La sécurité": "Securite",
    "L'offre de santé ": "Sante",
    "Votre situation professionnelle": "SituationPro",
    "Vos revenus": "Revenus",
    "La répartition de votre temps entre travail et temps personnel": "EquilibreViePro",
    "Votre logement": "Logement",
    "L'offre de services autour de chez vous": "ServicesProximite",
    "Votre accès à la culture": "Culture",
    "Vous sentez-vous bien entouré ?": "Entourage",
    "Vous sentez-vous impliqué dans la vie locale de votre commune ?": "ImplicationLocale",
    "Disponibilité des médecins généralistes": "MedecinsGeneralistes",
    "Temps d'attente pour avoir un rendez-vous": "AttenteRDV",
    "Disponibilité des médecins spécialistes": "MedecinsSpecialistes",
    "Etat du réseau routier": "ReseauRoutier",
    "Offre de transports en commun": "TransportsCommun",
    "Encombrement des voies du à la circulation": "Encombrement",
    "Sur une échelle de 1 à 5, pourriez-vous estimer à quel point vous êtes heureux ces derniers temps ?": "Bonheur_1_5",
    "Sur une échelle de 1 à 5,\xa0pourriez-vous évaluer votre qualité de vie ces derniers temps ?": "QdV_1_5",
    "Sur une échelle de 1 à 5, pourriez-vous évaluer votre confiance en l'avenir ?": "Confiance_1_5",
}


def _get_commune(row) -> str:
    c1 = row.get("Dans quelle commune résidez-vous ? ( A à S)", "")
    c2 = row.get("Dans quelle commune résidez-vous ? (T à Z)", "")
    if pd.notna(c1) and str(c1).strip():
        return str(c1).strip()
    if pd.notna(c2) and str(c2).strip():
        return str(c2).strip()
    return "Inconnue"


def load_enquete_responses(filepath: str = "./sortie_questionnaire_traited.csv") -> Tuple[List[str], List[Dict]]:
    df = pd.read_csv(filepath)
    texts, metas = [], []

    for _, row in df.iterrows():
        commune = _get_commune(row)
        genre = str(row.get("genre", "")).strip()
        age = str(row.get("Age", row.get("age", ""))).strip()
        csp = str(row.get("situation socioprofessionnelle", "")).strip()
        cat_age = str(row.get("Catégorie age", "")).strip()

        # En-tête du profil
        lines = [
            f"Répondant au questionnaire QdV Corse ({commune}, {genre}, {age} ans, {csp}, tranche {cat_age})."
        ]

        # Scores évalués
        scores = []
        for col, label in SCORE_COLS.items():
            val = row.get(col)
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                scores.append(f"{label}={val}")
        if scores:
            lines.append("Évaluations : " + ", ".join(scores) + ".")

        # Justifications images (verbatims courts)
        for col in [
            "Justifiez brièvement le choix de votre première\xa0image (5 mots).",
            "Justifiez brièvement le choix de votre deuxième image (5 mots).",
            "Justifiez brièvement le choix de votre troisième image (5 mots).",
        ]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                lines.append(f"Priorité citée : {str(val).strip()}.")

        text = " ".join(lines)
        texts.append(text)
        metas.append({
            "commune": commune,
            "nom": commune,
            "genre": genre,
            "age": str(age),
            "csp": csp,
            "cat_age": cat_age,
            "source_type": "enquete_repondant",
            "data_type": "quanti",
            "respondent_id": int(row.get("ID", 0)),
        })

    print(f"  {len(texts)} répondants chargés")
    return texts, metas


def index_enquete_responses(client, model):
    col = client.get_or_create_collection("enquete_responses")
    existing = set(col.get(include=[])["ids"])
    print(f"Collection enquete_responses : {col.count()} docs existants")

    texts, metas = load_enquete_responses()
    ids = [f"enquete_rep_{m['respondent_id']}" for m in metas]

    to_add = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if id_ not in existing]
    if not to_add:
        print("  Rien à ajouter")
        return

    ids_a, texts_a, metas_a = zip(*to_add)
    embeddings = embed_batch(model, list(texts_a))
    col.add(ids=list(ids_a), documents=list(texts_a), embeddings=embeddings, metadatas=list(metas_a))
    print(f"  [OK] {len(ids_a)} répondants ajoutés -> total {col.count()}")


# ============================================================
# 3. Scores moyens + dimensions par commune
# ============================================================

def load_enquete_scores_commune(
    scores_path: str = "./df_mean_by_commune.csv",
    dims_path: str = "./dimension_counts.csv",
) -> Tuple[List[str], List[Dict]]:

    df_scores = pd.read_csv(scores_path)
    df_dims = pd.read_csv(dims_path)

    # Top 3 dimensions par commune
    top_dims = (
        df_dims.sort_values(["commune", "percentage"], ascending=[True, False])
        .groupby("commune")
        .head(3)
        .groupby("commune")["dimensions_qdv"]
        .apply(list)
        .to_dict()
    )

    texts, metas = [], []
    score_cols = [
        "Score bonheur", "Score qualité de vie", "Score confiance avenir",
        "Transports", "Éducation", "Réseaux téléphoniques", "Institutions",
        "Tourisme", "Sécurité", "Santé", "Situation pro", "Revenus",
        "Temps travail/perso", "Logement", "Services locaux", "Culture",
        "Soutien social", "Vie associative",
    ]

    for _, row in df_scores.iterrows():
        commune = str(row.get("commune", "")).strip()
        n = int(row.get("total_respondants", 0))

        lines = [
            f"Scores moyens de l'enquête QdV pour la commune de {commune} ({n} répondants)."
        ]

        # Scores globaux
        for col in ["Score bonheur", "Score qualité de vie", "Score confiance avenir"]:
            val = row.get(col)
            if pd.notna(val):
                lines.append(f"{col} moyen : {val:.2f}/5.")

        # Scores par dimension
        dim_scores = []
        for col in score_cols[3:]:
            val = row.get(col)
            if pd.notna(val):
                dim_scores.append(f"{col}={val:.2f}")
        if dim_scores:
            lines.append("Scores moyens par dimension (1-5) : " + ", ".join(dim_scores) + ".")

        # Top dimensions citées comme prioritaires
        top = top_dims.get(commune, [])
        top_clean = [str(d) for d in top if pd.notna(d) and str(d).strip()]
        if top_clean:
            lines.append(f"Dimensions les plus citées comme prioritaires : {', '.join(top_clean)}.")

        text = " ".join(lines)
        texts.append(text)
        metas.append({
            "commune": commune,
            "nom": commune,
            "n_repondants": n,
            "source_type": "enquete_commune",
            "data_type": "quanti",
        })

    print(f"  {len(texts)} communes chargées")
    return texts, metas


def index_enquete_scores_commune(client, model):
    col = client.get_or_create_collection("enquete_scores_commune")
    existing = set(col.get(include=[])["ids"])
    print(f"Collection enquete_scores_commune : {col.count()} docs existants")

    texts, metas = load_enquete_scores_commune()
    ids = [f"enquete_commune_{m['commune'].lower().replace(' ', '_').replace('-', '_')}" for m in metas]

    to_add = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if id_ not in existing]
    if not to_add:
        print("  Rien à ajouter")
        return

    ids_a, texts_a, metas_a = zip(*to_add)
    embeddings = embed_batch(model, list(texts_a))
    col.add(ids=list(ids_a), documents=list(texts_a), embeddings=embeddings, metadatas=list(metas_a))
    print(f"  [OK] {len(ids_a)} communes ajoutées -> total {col.count()}")


# ============================================================
# 4. Wikipedia communes
# ============================================================

def load_communes_wiki(filepath: str = "./communes_corse_wikipedia.csv") -> Tuple[List[str], List[Dict]]:
    df = pd.read_csv(filepath)
    texts, metas = [], []

    for _, row in df.iterrows():
        commune = str(row.get("commune", "")).strip()
        contenu = str(row.get("contenu_wiki", row.get("résumé", ""))).strip()
        if not contenu or contenu == "nan":
            continue

        text = f"Informations sur la commune de {commune} (source Wikipedia) :\n{contenu}"
        texts.append(text)
        metas.append({
            "commune": commune,
            "nom": commune,
            "source_type": "wikipedia",
            "data_type": "mixte",
        })

    print(f"  {len(texts)} communes wiki chargées")
    return texts, metas


def index_communes_wiki(client, model):
    col = client.get_or_create_collection("communes_wiki")
    existing = set(col.get(include=[])["ids"])
    print(f"Collection communes_wiki : {col.count()} docs existants")

    texts, metas = load_communes_wiki()
    ids = [f"wiki_{m['commune'].lower().replace(' ', '_').replace('-', '_')}" for m in metas]

    to_add = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if id_ not in existing]
    if not to_add:
        print("  Rien à ajouter")
        return

    ids_a, texts_a, metas_a = zip(*to_add)
    embeddings = embed_batch(model, list(texts_a))
    col.add(ids=list(ids_a), documents=list(texts_a), embeddings=embeddings, metadatas=list(metas_a))
    print(f"  [OK] {len(ids_a)} communes ajoutées -> total {col.count()}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="all",
                        choices=["all", "entretiens", "enquete", "wiki"])
    args = parser.parse_args()

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    model = get_embed_model()

    print("\n" + "=" * 60)

    if args.collection in ("all", "entretiens"):
        print("\n[1/4] Entretiens semi-directifs -> portrait_entretiens")
        index_entretiens(client, model)

    if args.collection in ("all", "enquete"):
        print("\n[2/4] Reponses questionnaire -> enquete_responses")
        index_enquete_responses(client, model)

        print("\n[3/4] Scores par commune -> enquete_scores_commune")
        index_enquete_scores_commune(client, model)

    if args.collection in ("all", "wiki"):
        print("\n[4/4] Wikipedia communes -> communes_wiki")
        index_communes_wiki(client, model)

    print("\n" + "=" * 60)
    print("INDEXATION TERMINÉE. Collections dans chroma_portrait :")
    for col in client.list_collections():
        c = client.get_collection(col.name)
        print(f"  {col.name} : {c.count()} docs")


if __name__ == "__main__":
    main()
