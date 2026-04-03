"""
RAPTOR-lite quantitatif : synthèses statistiques pré-calculées sur données questionnaire.

Génère des résumés structurés par groupes démographiques (âge, CSP, commune, et
combinaisons 2D), stockés dans ChromaDB — miroir du RAPTOR quali (rag_v9_raptor.py)
mais sur les données quantitatives du questionnaire.

Les synthèses sont calculées algorithmiquement (pas de LLM) : moyennes, distributions,
classements. Elles sont ensuite encodées avec BGE-M3 pour retrieval sémantique.

Usage:
    python rag_quanti_raptor.py --build          # Génère toutes les synthèses
    python rag_quanti_raptor.py --stats          # Affiche les stats par vue
    python rag_quanti_raptor.py --query "..."    # Test de retrieval
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Constantes
# ============================================================

CSV_PATH = "./donnees_brutes/sortie_questionnaire_traited.csv"
CHROMA_PATH = "./chroma_portrait"
TARGET_COLLECTION = "raptor_quanti_summaries"
EMBED_MODEL_NAME = "BAAI/bge-m3"
MATERIALIZATION_THRESHOLD = 5   # N min de répondants pour matérialiser une vue

VIEW_DEFINITIONS = [
    # 2D (specificité 2, testées en premier)
    {"name": "age_range*profession", "dimensions": ["age_range", "profession"], "specificity": 2},
    {"name": "age_range*commune",    "dimensions": ["age_range", "commune"],    "specificity": 2},
    {"name": "profession*commune",   "dimensions": ["profession", "commune"],   "specificity": 2},
    # 1D (fallback)
    {"name": "age_range",  "dimensions": ["age_range"],  "specificity": 1},
    {"name": "profession", "dimensions": ["profession"], "specificity": 1},
    {"name": "commune",    "dimensions": ["commune"],    "specificity": 1},
]

# Scores numériques pour les échelles de satisfaction
SATISFACTION_MAP = {
    "Très satisfait":     5,
    "Satisfait":          4,
    "Neutre":             3,
    "Peu satisfait":      2,
    "Très peu satisfait": 1,
}
ENTOURAGE_MAP = {
    "Très bien entouré":  5,
    "Bien entouré":       4,
    "Moyennement entouré":3,
    "Peu entouré":        2,
    "Très peu entouré":   1,
}
IMPLICATION_MAP = {
    "Très impliqué":         5,
    "Impliqué":              4,
    "Moyennement Impliqué":  3,
    "Peu impliqué":          2,
    "Très peu impliqué":     1,
}

# Colonnes de satisfaction principales (col → label court)
SATISFACTION_COLS = {
    "Les services de transports":                                         "Transports",
    "L'accès à l'éducation":                                              "Éducation",
    "La couverture des réseaux téléphoniques":                            "Réseaux",
    "Les institutions étatiques (niveau de confiance)":                   "Institutions",
    "Le tourisme (ressenti localement)":                                  "Tourisme",
    "La sécurité":                                                        "Sécurité",
    "L'offre de santé ":                                                  "Santé",
    "Votre situation professionnelle":                                    "Emploi",
    "Vos revenus":                                                        "Revenus",
    "La répartition de votre temps entre travail et temps personnel":     "Équilibre vie pro/perso",
    "Votre logement":                                                     "Logement",
    "L'offre de services autour de chez vous":                            "Services de proximité",
    "Votre accès à la culture":                                           "Culture",
}

COL_BONHEUR    = "Sur une échelle de 1 à 5, pourriez-vous estimer à quel point vous êtes heureux ces derniers temps ?"
COL_QDV        = "Sur une échelle de 1 à 5, pourriez-vous évaluer votre qualité de vie ces derniers temps ?"
COL_CONFIANCE  = "Sur une échelle de 1 à 5, pourriez-vous évaluer votre confiance en l'avenir ?"
COL_ENTOURAGE  = "Vous sentez-vous bien entouré ?"
COL_IMPLICATION = "Vous sentez-vous impliqué dans la vie locale de votre commune ?"
COL_DIM_CHOICES = "Pour vous, qu'est-ce qui est important pour votre qualité de vie ? Choisissez 3 images"

# Dimensions QdV reconnues (nommées, pas "Option X")
QDV_DIMENSIONS = [
    "Environnement", "Santé", "Culture", "Éducation", "Education",
    "Emploi", "Logement", "Revenus", "Transports", "Sécurité",
    "Tourisme", "Communauté et relations", "Réseau", "Services de proximité",
    "Confiance envers les institutions", "Démographie",
    "Ratio vie pro/ vie perso", "Vie pro/perso",
]


# ============================================================
# Chargement et nettoyage des données
# ============================================================

def load_data() -> pd.DataFrame:
    """Charge le CSV et normalise les colonnes clés."""
    df = pd.read_csv(CSV_PATH, index_col=0)

    # Fusion des deux colonnes commune
    col_a = "Dans quelle commune résidez-vous ? ( A à S)"
    col_b = "Dans quelle commune résidez-vous ? (T à Z)"
    df["commune"] = (df[col_a].fillna("") + df[col_b].fillna("")).str.strip()

    # Renommage des colonnes de dimension
    df["age_range"]  = df["Catégorie age"].str.strip()
    df["profession"] = df["situation socioprofessionnelle"].str.strip()

    # Nettoyage : supprimer lignes avec age_range invalide ou '0'
    df = df[df["age_range"].isin(["15-29", "30-44", "45-59", "60-74", "75+"])]

    # Supprimer lignes où profession contient un nom de commune (erreur de saisie)
    valid_professions = set(IMPLICATION_MAP.keys()) | {
        "Agriculteur(trice), artisan(e) ou commerçant(e)", "Autre", "Fonctionnaire",
        "Retraité(e)", "Salarié(e) – Cadre ou profession intermédiaire",
        "Salarié(e) – Employé(e)", "Sans emploi (en recherche d'emploi)",
        "Travailleur(se) indépendant(e) / Entrepreneur(e)", "Étudiant(e)",
    }
    # Garder les lignes dont la profession est dans la liste connue
    df = df[df["profession"].notna()]

    # Score numérique pour les colonnes satisfaction
    for col in SATISFACTION_COLS:
        if col in df.columns:
            df[f"_score_{col}"] = df[col].map(SATISFACTION_MAP)

    # Scores entourage et implication
    if COL_ENTOURAGE in df.columns:
        df["_score_entourage"]   = df[COL_ENTOURAGE].map(ENTOURAGE_MAP)
    if COL_IMPLICATION in df.columns:
        df["_score_implication"] = df[COL_IMPLICATION].map(IMPLICATION_MAP)

    return df


# ============================================================
# Génération de synthèse statistique (algorithmique)
# ============================================================

def _star(score: Optional[float]) -> str:
    """Convertit un score /5 en indicateur textuel."""
    if score is None:
        return "n/d"
    if score >= 4.2:
        return f"{score:.2f}/5 ✦✦ (fort)"
    if score >= 3.3:
        return f"{score:.2f}/5 ✦ (moyen-fort)"
    if score >= 2.5:
        return f"{score:.2f}/5 → (moyen)"
    return f"{score:.2f}/5 ↓ (faible)"


def _mean_or_none(series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(vals.mean()), 2) if len(vals) > 0 else None


def _pct_dist(series, score_map: Dict[str, int]) -> List[Tuple[str, float]]:
    """Retourne la distribution en % de chaque modalité, triée par fréquence desc."""
    total = series.notna().sum()
    if total == 0:
        return []
    counts = series.value_counts(dropna=True)
    return [
        (label, round(100 * cnt / total, 1))
        for label, cnt in counts.items()
    ]


def build_summary(group_df: pd.DataFrame, dim_label: str) -> str:
    """
    Génère la synthèse statistique textuelle pour un groupe.
    Entièrement algorithmique — pas de LLM.
    """
    n = len(group_df)
    lines = []

    lines.append(f"## Groupe : {dim_label} (N={n})")
    lines.append("")

    # ── Bien-être global ──────────────────────────────────────
    bonheur   = _mean_or_none(group_df[COL_BONHEUR])  if COL_BONHEUR   in group_df.columns else None
    qdv       = _mean_or_none(group_df[COL_QDV])      if COL_QDV       in group_df.columns else None
    confiance = _mean_or_none(group_df[COL_CONFIANCE]) if COL_CONFIANCE in group_df.columns else None

    lines.append("### Bien-être global")
    lines.append(f"- Bonheur : {_star(bonheur)}")
    lines.append(f"- Qualité de vie : {_star(qdv)}")
    lines.append(f"- Confiance en l'avenir : {_star(confiance)}")
    lines.append("")

    # ── Dimensions prioritaires ────────────────────────────────
    if COL_DIM_CHOICES in group_df.columns:
        dim_counter: Dict[str, int] = defaultdict(int)
        for val in group_df[COL_DIM_CHOICES].dropna():
            parts = [p.strip() for p in str(val).split(";") if p.strip()]
            for p in parts:
                # Ne compter que les dimensions nommées (pas "Option X")
                if not p.startswith("Option") and len(p) > 2:
                    dim_counter[p] += 1

        if dim_counter:
            lines.append("### Dimensions prioritaires (% de citations parmi top 3 choix)")
            top_dims = sorted(dim_counter.items(), key=lambda x: -x[1])[:8]
            for dim, cnt in top_dims:
                pct = round(100 * cnt / n, 1)
                lines.append(f"- {dim} : {pct}%")
            lines.append("")

    # ── Satisfaction par domaine ──────────────────────────────
    scores: List[Tuple[str, float]] = []
    for col, label in SATISFACTION_COLS.items():
        score_col = f"_score_{col}"
        if score_col in group_df.columns:
            m = _mean_or_none(group_df[score_col])
            if m is not None:
                scores.append((label, m))

    if scores:
        scores.sort(key=lambda x: -x[1])
        lines.append("### Satisfaction par domaine (score moyen /5)")
        for label, m in scores:
            lines.append(f"- {label} : {_star(m)}")
        lines.append("")

        strong = [l for l, m in scores if m >= 3.5]
        weak   = [l for l, m in scores if m <= 2.5]
        if strong:
            lines.append(f"**Points forts** (≥ 3.5/5) : {', '.join(strong)}")
        if weak:
            lines.append(f"**Points faibles** (≤ 2.5/5) : {', '.join(weak)}")
        if strong or weak:
            lines.append("")

    # ── Lien social et implication ────────────────────────────
    entourage_score   = _mean_or_none(group_df["_score_entourage"])   if "_score_entourage"   in group_df.columns else None
    implication_score = _mean_or_none(group_df["_score_implication"]) if "_score_implication" in group_df.columns else None

    if entourage_score is not None or implication_score is not None:
        lines.append("### Lien social")
        if entourage_score is not None:
            lines.append(f"- Sentiment d'entourage : {_star(entourage_score)}")
        if implication_score is not None:
            lines.append(f"- Implication dans la vie locale : {_star(implication_score)}")
        lines.append("")

    # ── Distribution satisfaction sur domaines clés ──────────
    key_cols = {
        "L'offre de santé ":  "Santé",
        "Les services de transports": "Transports",
        "Votre logement": "Logement",
    }
    dist_parts = []
    for col, label in key_cols.items():
        if col in group_df.columns:
            dist = _pct_dist(group_df[col], SATISFACTION_MAP)
            if dist:
                top2 = ", ".join(f"{v} ({p}%)" for v, p in dist[:2])
                dist_parts.append(f"{label} : {top2}")
    if dist_parts:
        lines.append("### Distribution des réponses (top 2 modalités)")
        for part in dist_parts:
            lines.append(f"- {part}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# QuantiRaptorBuilder
# ============================================================

class QuantiRaptorBuilder:
    """Génère les vues RAPTOR quantitatives et les stocke dans ChromaDB."""

    def __init__(self,
                 csv_path: str = CSV_PATH,
                 chroma_path: str = CHROMA_PATH,
                 target_collection: str = TARGET_COLLECTION,
                 min_respondents: int = MATERIALIZATION_THRESHOLD):
        self.csv_path = csv_path
        self.chroma_path = chroma_path
        self.target_collection_name = target_collection
        self.min_respondents = min_respondents

    def build_all(self, incremental: bool = False) -> Dict:
        """Génère toutes les vues RAPTOR quanti. Retourne les statistiques."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("=" * 70)
        print("RAPTOR-lite quantitatif : Construction des vues analytiques")
        if incremental:
            print("  Mode INCREMENTAL : seules les synthèses manquantes seront générées")
        print("=" * 70)

        df = load_data()
        print(f"Données chargées : {len(df)} répondants valides")

        client = chromadb.PersistentClient(path=self.chroma_path)

        if incremental:
            target = client.get_or_create_collection(
                self.target_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            existing_ids = set(target.get()["ids"])
            print(f"Collection cible : {self.target_collection_name} ({len(existing_ids)} synthèses existantes)")
        else:
            try:
                client.delete_collection(self.target_collection_name)
            except Exception:
                pass
            target = client.create_collection(
                self.target_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            existing_ids = set()
            print(f"Collection cible : {self.target_collection_name} (recréée)")

        print(f"Chargement du modèle d'embeddings ({EMBED_MODEL_NAME})...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        stats = {"total_views": 0, "total_summaries": 0, "by_view": {}}

        for view_def in VIEW_DEFINITIONS:
            print(f"\n--- Vue : {view_def['name']} ---")
            n = self._build_view(view_def, df, target, embed_model, existing_ids)
            stats["by_view"][view_def["name"]] = n
            stats["total_summaries"] += n
            stats["total_views"] += 1

        print(f"\n{'=' * 70}")
        print(f"BUILD TERMINÉ : {stats['total_summaries']} synthèses générées")
        for view_name, count in stats["by_view"].items():
            print(f"  {view_name}: {count}")
        print("=" * 70)

        return stats

    def _build_view(self, view_def: Dict, df: pd.DataFrame,
                    target_collection, embed_model, existing_ids: set) -> int:
        """Groupe et matérialise une vue. Retourne le nombre de groupes."""
        dims = view_def["dimensions"]

        # Grouper les répondants par les dimensions de la vue
        groups: Dict[Tuple, pd.DataFrame] = {}
        for key_values, group_df in df.groupby(dims):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            if all(v and str(v).strip() for v in key_values):
                groups[key_values] = group_df

        valid_groups = {k: v for k, v in groups.items() if len(v) >= self.min_respondents}
        print(f"  {len(groups)} groupes, {len(valid_groups)} avec >= {self.min_respondents} repondants")

        materialized = 0
        for key_values, group_df in sorted(valid_groups.items()):
            dim_values = {d: v for d, v in zip(dims, key_values)}
            doc_id = _make_doc_id(view_def["name"], dim_values)

            if doc_id in existing_ids:
                print(f"  [SKIP] {dim_values}")
                materialized += 1
                continue

            dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items())
            summary = build_summary(group_df, dim_label)

            embedding = embed_model.encode(f"passage: {summary}").tolist()

            meta = {
                "view_name":   view_def["name"],
                "specificity": view_def["specificity"],
                "dim1_name":   dims[0],
                "dim1_value":  str(key_values[0]),
                "dim2_name":   dims[1] if len(dims) > 1 else "",
                "dim2_value":  str(key_values[1]) if len(key_values) > 1 else "",
                "num_respondents": int(len(group_df)),
                "built_at":    datetime.now().isoformat(),
                "data_source": "questionnaire_quanti",
            }

            target_collection.add(
                ids=[doc_id],
                documents=[summary],
                embeddings=[embedding],
                metadatas=[meta]
            )
            materialized += 1
            print(f"  [OK] {dim_label} (N={len(group_df)})")

        return materialized


def _make_doc_id(view_name: str, dim_values: Dict[str, str]) -> str:
    slug = "_".join(
        v.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("–", "-")[:30]
        for v in dim_values.values()
    )
    return f"quanti_raptor_{view_name}_{slug}"


# ============================================================
# QuantiRaptorRetriever
# ============================================================

class QuantiRaptorRetriever:
    """Retrieval RAPTOR-lite sur données quantitatives."""

    # Mots-clés pour détecter l'âge dans une question
    _AGE_KEYWORDS = {
        "15-29": ["jeune", "jeunes", "adolescent", "étudiant", "lycéen", "15-29"],
        "30-44": ["30-44", "30 ans", "adulte", "trentenaire", "quadra"],
        "45-59": ["45-59", "45 ans", "cinquantaine", "milieu de vie"],
        "60-74": ["60-74", "senior", "retraité", "aîné", "60 ans"],
        "75+":   ["75", "très âgé", "grand âge"],
    }

    _PROFESSION_KEYWORDS = {
        "Étudiant(e)": ["étudiant", "étudiante", "lycéen", "étude"],
        "Fonctionnaire": ["fonctionnaire", "agent public", "territorial"],
        "Retraité(e)": ["retraité", "retraités", "pension", "senior"],
        "Salarié(e) – Employé(e)": ["employé", "salarié", "ouvrier"],
        "Salarié(e) – Cadre ou profession intermédiaire": ["cadre", "ingénieur", "manager", "profession intermédiaire"],
        "Travailleur(se) indépendant(e) / Entrepreneur(e)": ["indépendant", "entrepreneur", "auto-entrepreneur"],
        "Agriculteur(trice), artisan(e) ou commerçant(e)": ["agriculteur", "artisan", "commerçant"],
    }

    def __init__(self,
                 chroma_path: str = CHROMA_PATH,
                 collection: str = TARGET_COLLECTION):
        self.chroma_path = chroma_path
        self.collection_name = collection
        self._embed_model = None
        self._col = None

    def init(self):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=self.chroma_path)
        self._col = client.get_collection(self.collection_name)
        print(f"QuantiRaptorRetriever : {self._col.count()} synthèses disponibles")

    def query(self, question: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Retourne (context_str, sources_list)."""
        detected_age  = self._detect_age(question)
        detected_prof = self._detect_profession(question)

        # Chercher la vue la plus spécifique
        filters = []
        if detected_age:
            filters.append({"dim1_value": {"$eq": detected_age}})
            if detected_prof:
                filters.append({"dim2_value": {"$eq": detected_prof}})
        elif detected_prof:
            filters.append({"dim1_value": {"$eq": detected_prof}})

        query_emb = self._embed_model.encode(f"query: {question}").tolist()

        where = None
        if len(filters) == 2:
            where = {"$and": filters}
        elif len(filters) == 1:
            where = filters[0]

        try:
            kwargs = {"query_embeddings": [query_emb], "n_results": k,
                      "include": ["documents", "metadatas", "distances"]}
            if where:
                kwargs["where"] = where
            res = self._col.query(**kwargs)
        except Exception:
            # Fallback sans filtre
            res = self._col.query(
                query_embeddings=[query_emb], n_results=k,
                include=["documents", "metadatas", "distances"]
            )

        docs   = res["documents"][0]
        metas  = res["metadatas"][0]
        dists  = res["distances"][0]

        context_parts = []
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            context_parts.append(
                f"[Synthèse quantitative — vue {meta.get('view_name', '?')}]\n{doc}"
            )
            sources.append({
                "rank": i + 1,
                "type": "raptor_quanti_summary",
                "view": meta.get("view_name"),
                "dim1": f"{meta.get('dim1_name')}={meta.get('dim1_value')}",
                "dim2": f"{meta.get('dim2_name')}={meta.get('dim2_value')}" if meta.get("dim2_value") else None,
                "num_respondents": meta.get("num_respondents"),
                "score": round(1 - dist, 3),
                "extrait": doc[:400],
            })

        return "\n\n".join(context_parts), sources

    def _detect_age(self, question: str) -> Optional[str]:
        q = question.lower()
        for age_range, keywords in self._AGE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return age_range
        return None

    def _detect_profession(self, question: str) -> Optional[str]:
        q = question.lower()
        for prof, keywords in self._PROFESSION_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return prof
        return None


# ============================================================
# CLI
# ============================================================

def cmd_build(args):
    builder = QuantiRaptorBuilder(min_respondents=args.min_n)
    builder.build_all(incremental=args.incremental)


def cmd_stats(args):
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(TARGET_COLLECTION)
    except Exception:
        print("Collection non trouvée. Lancez --build d'abord.")
        return

    all_data = col.get(include=["metadatas"])
    by_view: Dict[str, int] = defaultdict(int)
    for meta in all_data["metadatas"]:
        by_view[meta.get("view_name", "?")] += 1

    print(f"\nCollection : {TARGET_COLLECTION}")
    print(f"Total synthèses : {col.count()}\n")
    print(f"{'Vue':<30} {'N synthèses':>12}")
    print("-" * 44)
    for view_name, count in sorted(by_view.items()):
        print(f"{view_name:<30} {count:>12}")


def cmd_query(args):
    retriever = QuantiRaptorRetriever()
    retriever.init()
    ctx, sources = retriever.query(args.question, k=args.k)
    print("\n=== CONTEXTE ===")
    print(ctx[:2000])
    print("\n=== SOURCES ===")
    for s in sources:
        print(f"  [{s['rank']}] {s['view']} — {s['dim1']}" + (f" × {s['dim2']}" if s.get('dim2') else "") +
              f" (N={s['num_respondents']}, score={s['score']})")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="RAPTOR-lite quantitatif")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Construire toutes les synthèses")
    p_build.add_argument("--incremental", action="store_true",
                         help="Ne régénère pas les synthèses existantes")
    p_build.add_argument("--min-n", type=int, default=MATERIALIZATION_THRESHOLD,
                         help=f"N min répondants (défaut: {MATERIALIZATION_THRESHOLD})")

    sub.add_parser("stats", help="Afficher les statistiques")

    p_query = sub.add_parser("query", help="Tester le retrieval")
    p_query.add_argument("question", type=str)
    p_query.add_argument("--k", type=int, default=3)

    # Permet syntaxe --build, --stats, --query (avec tirets)
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        sys.argv[1] = sys.argv[1].lstrip("-")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()
