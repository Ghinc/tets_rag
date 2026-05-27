"""
build_classement_scores.py

Construit des documents de classement des dimensions de satisfaction
à partir des données brutes de l'enquête citoyenne.

Ces documents permettent de répondre aux questions de type :
  - "Quelle dimension obtient la note la plus faible ?"
  - "Quel est le classement des dimensions à Ajaccio ?"
  - "Pour les retraités, quelle dimension est la plus satisfaisante ?"

Scopes produits :
  1. global         — tous répondants (N~246)
  2. commune        — par commune (68 docs)
  3. cat_age        — par tranche d'âge (4 docs)
  4. csp            — par CSP (9 docs)

Total : ~84 documents, upsertés dans enquete_scores_commune.

Usage :
    python build_classement_scores.py
    python build_classement_scores.py --dry-run    # affiche sans upsert
    python build_classement_scores.py --stats      # stats collection existante
"""

import os
import sys
import argparse
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Forcer UTF-8 sur Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.abspath("./model_cache"))

ENQUETE_CSV    = "./sortie_questionnaire_traited.csv"
CHROMA_PATH    = "./chroma_portrait"
COLLECTION     = "enquete_scores_commune"
EMBED_MODEL    = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# Seuil minimum de répondants pour calculer un classement
MIN_N = 3


# ── Normalisation ────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Supprime les accents et met en minuscules pour la recherche de colonnes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _find_col(columns: List[str], keyword: str) -> Optional[str]:
    """Retourne la première colonne dont le nom normalisé contient `keyword`."""
    kw_n = _norm(keyword)
    for col in columns:
        if kw_n in _norm(col):
            return col
    return None


def _find_cols(columns: List[str], keywords: List[str]) -> List[str]:
    """Pour chaque keyword, trouve la première colonne correspondante."""
    result = []
    for kw in keywords:
        col = _find_col(columns, kw)
        if col and col not in result:
            result.append(col)
    return result


# ── Mapping dimensions → colonnes CSV ────────────────────────────────────────
# Chaque dimension liste des keywords qui permettent de retrouver
# les colonnes du CSV correspondantes (matching par sous-chaîne après norm).

DIM_COL_KEYWORDS: Dict[str, List[str]] = {
    "Transports":             [
        "services de transport",     # "Les services de transports"
        "transports en commun",      # "Offre de transports en commun"
        "seau routier",              # "Etat du réseau routier"
        "encombrement",              # "Encombrement des voies..."
    ],
    "Santé":                  [
        "offre de sant",             # "L'offre de santé "
        "decins generalistes",       # "Disponibilité des médecins généralistes"
        "attente pour avoir",        # "Temps d'attente pour avoir un rendez-vous"
        "decins sp",                 # "Disponibilité des médecins spécialistes"
    ],
    "Éducation":              ["ducation"],      # "L'accès à l'éducation"
    "Logement":               ["votre logement"],
    "Revenus":                ["vos revenus"],
    "Emploi":                 ["votre situation professionnelle"],
    "Sécurité":               ["curit"],         # "La sécurité"
    "Culture":                ["la culture"],    # "Votre accès à la culture"
    "Services de proximité":  ["services autour de chez vous"],
    "Réseau":                 ["couverture"],    # "La couverture des réseaux téléphoniques"
    "Ratio vie pro/vie perso":["partition de votre temps"],  # "La répartition de votre temps..."
    "Communauté et relations":["entour", "impliqu"],  # "Vous sentez-vous bien entouré ?"
    "Tourisme":               ["tourisme"],
    "Institutions":           ["institutions"],
}

# Échelle de satisfaction standard (Très peu → Très satisfait)
LIKERT_MAP: Dict[str, float] = {
    "Très satisfait":     5.0,
    "Satisfait":          4.0,
    "Neutre":             3.0,
    "Peu satisfait":      2.0,
    "Très peu satisfait": 1.0,
}

# Échelle pour les colonnes "Communauté et relations" (label différent)
LIKERT_COMMUNITY_MAP: Dict[str, float] = {
    # Entourage
    "Très bien entouré":   5.0,
    "Bien entouré":        4.0,
    "Moyennement entouré": 3.0,
    "Peu entouré":         2.0,
    "Très peu entouré":    1.0,
    # Implication locale
    "Très impliqué":         5.0,
    "Impliqué":              4.0,
    "Moyennement Impliqué":  3.0,
    "Peu impliqué":          2.0,
    "Très peu impliqué":     1.0,
}

# Colonnes "Communauté" pour déterminer si on utilise la table community map
_COMMUNITY_KEYWORDS = ["entour", "impliqu"]


def _to_num(val, community: bool = False) -> Optional[float]:
    """Convertit une valeur catégorielle en score numérique."""
    if not isinstance(val, str):
        return None
    v = val.strip().rstrip("\xa0").strip()
    if community:
        combined = {**LIKERT_MAP, **LIKERT_COMMUNITY_MAP}
        return combined.get(v) or LIKERT_COMMUNITY_MAP.get(v)
    # Gérer "Moyennement satisfait" comme Neutre (3)
    if "moyennement" in v.lower():
        return 3.0
    return LIKERT_MAP.get(v)


# ── Interprétation textuelle du score ────────────────────────────────────────

def _level(score: float) -> str:
    if score >= 4.5: return "très élevé"
    if score >= 3.5: return "moyen-fort"
    if score >= 2.5: return "moyen"
    if score >= 1.5: return "faible"
    return "très faible"


# ── Calcul des scores par groupe ─────────────────────────────────────────────

def _build_col_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Résout le mapping dimension → colonnes réelles du DataFrame."""
    resolved: Dict[str, List[str]] = {}
    for dim, keywords in DIM_COL_KEYWORDS.items():
        cols = _find_cols(list(df.columns), keywords)
        if cols:
            resolved[dim] = cols
        else:
            print(f"  [WARN] Aucune colonne trouvée pour dimension '{dim}'")
    return resolved


def compute_dim_scores(
    df_group: pd.DataFrame,
    col_map: Dict[str, List[str]],
) -> List[Tuple[str, float]]:
    """
    Retourne [(dim_label, mean_score), ...] trié décroissant (meilleur en tête).
    Seules les dimensions avec au moins 1 valeur non-nulle sont incluses.
    """
    scores = []
    for dim, cols in col_map.items():
        is_community = dim == "Communauté et relations"
        vals: List[float] = []
        for col in cols:
            if col not in df_group.columns:
                continue
            for v in df_group[col]:
                num = _to_num(v, community=is_community)
                if num is not None:
                    vals.append(num)
        if vals:
            scores.append((dim, round(sum(vals) / len(vals), 2)))
    return sorted(scores, key=lambda x: -x[1])


# ── Formatage du texte ────────────────────────────────────────────────────────

def format_classement(ranked: List[Tuple[str, float]], scope_label: str, n: int) -> str:
    lines = [
        f"Classement des {len(ranked)} dimensions de satisfaction — {scope_label} (N={n} répondants).",
        "",
        "Du plus satisfaisant au moins satisfaisant :",
    ]
    for i, (dim, score) in enumerate(ranked, 1):
        label = _level(score)
        lines.append(f"  {i:2d}. {dim:<30s} : {score:.2f}/5 ({label})")
    lines += [
        "",
        f"-> NOTE LA PLUS FAIBLE   : {ranked[-1][0]} ({ranked[-1][1]:.2f}/5)",
        f"-> NOTE LA PLUS ELEVEE   : {ranked[0][0]} ({ranked[0][1]:.2f}/5)",
        f"-> ECART min/max         : {ranked[0][1] - ranked[-1][1]:.2f} points",
    ]
    return "\n".join(lines)


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embed_model():
    from sentence_transformers import SentenceTransformer
    print("Chargement BGE-M3...")
    return SentenceTransformer(EMBED_MODEL)


def embed_batch(model, texts: List[str], prefix: str = "passage: ") -> List[List[float]]:
    return [model.encode(prefix + t).tolist() for t in texts]


# ── Génération des documents ──────────────────────────────────────────────────

def build_all_docs(df: pd.DataFrame) -> List[Tuple[str, str, dict]]:
    """
    Retourne une liste de (doc_id, text, metadata) pour tous les scopes.
    """
    # Résoudre le mapping colonnes une seule fois
    col_map = _build_col_map(df)
    docs: List[Tuple[str, str, dict]] = []

    # ── 1. Global ─────────────────────────────────────────────────────────────
    df_valid = df.dropna(subset=[col_map["Éducation"][0]] if "Éducation" in col_map else [])
    n_global = len(df)
    ranked = compute_dim_scores(df, col_map)
    if ranked:
        text = format_classement(ranked, "Corse entière", n_global)
        docs.append((
            "classement_dimensions_global",
            text,
            {"source_type": "classement", "scope": "global"},
        ))
        print(f"  [global] {len(ranked)} dimensions, N={n_global}")

    # ── 2. Par commune ────────────────────────────────────────────────────────
    if "commune" in df.columns:
        for commune, grp in df.groupby("commune"):
            if len(grp) < MIN_N:
                continue
            ranked = compute_dim_scores(grp, col_map)
            if not ranked:
                continue
            scope_label = f"commune de {commune}"
            text = format_classement(ranked, scope_label, len(grp))
            doc_id = f"classement_dimensions_commune_{commune.lower().replace(' ', '_').replace('-', '_')}"
            docs.append((
                doc_id,
                text,
                {"source_type": "classement", "scope": "commune", "commune": commune, "nom": commune},
            ))
        n_communes = sum(1 for d in docs if d[2].get("scope") == "commune")
        print(f"  [commune] {n_communes} communes (seuil N>={MIN_N})")

    # ── 3. Par tranche d'âge ──────────────────────────────────────────────────
    cat_age_col = _find_col(list(df.columns), "tegorie age")
    if cat_age_col:
        for cat_age, grp in df.groupby(cat_age_col):
            if len(grp) < MIN_N:
                continue
            ranked = compute_dim_scores(grp, col_map)
            if not ranked:
                continue
            scope_label = f"tranche d'âge {cat_age}"
            text = format_classement(ranked, scope_label, len(grp))
            cat_slug = str(cat_age).replace("-", "_").replace("/", "_").replace(" ", "_")
            doc_id = f"classement_dimensions_cat_age_{cat_slug}"
            docs.append((
                doc_id,
                text,
                {"source_type": "classement", "scope": "cat_age", "cat_age": str(cat_age)},
            ))
        n_ages = sum(1 for d in docs if d[2].get("scope") == "cat_age")
        print(f"  [cat_age] {n_ages} tranches d'âge")
    else:
        print("  [WARN] Colonne 'Catégorie age' introuvable")

    # ── 4. Par CSP ────────────────────────────────────────────────────────────
    csp_col = _find_col(list(df.columns), "socio")
    if csp_col:
        for csp, grp in df.groupby(csp_col):
            if len(grp) < MIN_N:
                continue
            ranked = compute_dim_scores(grp, col_map)
            if not ranked:
                continue
            scope_label = f"CSP « {csp} »"
            text = format_classement(ranked, scope_label, len(grp))
            csp_slug = (str(csp)
                        .lower()
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("/", "_")
                        .replace("'", "")
                        .replace(",", "")
                        .replace("-", "_"))
            doc_id = f"classement_dimensions_csp_{csp_slug}"
            docs.append((
                doc_id,
                text,
                {"source_type": "classement", "scope": "csp", "csp": str(csp)},
            ))
        n_csps = sum(1 for d in docs if d[2].get("scope") == "csp")
        print(f"  [csp] {n_csps} CSP")
    else:
        print("  [WARN] Colonne CSP introuvable")

    return docs


# ── Classements des communes (inverse : commune → score sur chaque dimension) ──

def _mean_numeric_col(df_group: pd.DataFrame, col: Optional[str]) -> Optional[float]:
    """Retourne la moyenne numérique d'une colonne 1-5 (questions globales)."""
    if not col or col not in df_group.columns:
        return None
    vals = []
    for v in df_group[col]:
        try:
            fv = float(v)
            if 1.0 <= fv <= 5.0:
                vals.append(fv)
        except (ValueError, TypeError):
            pass
    return round(sum(vals) / len(vals), 2) if vals else None


def format_commune_ranking(
    ranked: List[Tuple[str, float, int]],  # (commune, score, n)
    criterion_label: str,
    note: str = "",
) -> str:
    lines = [
        f"Classement des communes — {criterion_label}",
        f"(Seuil minimum {MIN_N} répondants par commune)",
        "",
        f"  {'Rang':<4} {'Commune':<26} {'N répondants':>13}   {'Score /5':>9}",
        "  " + "─" * 60,
    ]
    for i, (commune, score, n) in enumerate(ranked, 1):
        lvl = _level(score)
        lines.append(f"  {i:2d}. {commune:<26} N={n:>3} répondants   {score:.2f}/5 ({lvl})")
    if ranked:
        best_c, best_s, best_n = ranked[0]
        worst_c, worst_s, worst_n = ranked[-1]
        lines += [
            "",
            f"-> COMMUNE LA MIEUX ÉVALUÉE   : {best_c} ({best_s:.2f}/5, N={best_n})",
            f"-> COMMUNE LA MOINS BIEN ÉVAL. : {worst_c} ({worst_s:.2f}/5, N={worst_n})",
        ]
    if note:
        lines += ["", f"Note : {note}"]
    return "\n".join(lines)


def build_commune_rankings(df: pd.DataFrame) -> List[Tuple[str, str, dict]]:
    """
    Construit les classements des communes par score perçu (inverse du classement dimensions).
    - classement_communes_global      : communes classées par score moyen toutes dims
    - classement_communes_bien_etre   : communes classées par bonheur + QdV
    - classement_communes_confiance_avenir : communes classées par confiance
    - classement_communes_dim_{slug}  : communes classées par chaque dimension (×14)
    """
    if "commune" not in df.columns:
        print("  [WARN] Colonne commune introuvable, skip commune rankings")
        return []

    col_map = _build_col_map(df)
    col_bonheur   = _find_col(list(df.columns), "heureux")
    col_qdv       = _find_col(list(df.columns), "qualite de vie")
    col_confiance = _find_col(list(df.columns), "confiance en l")

    # Calculer scores par commune
    commune_scores: Dict[str, dict] = {}
    for commune, grp in df.groupby("commune"):
        if not commune or str(commune).strip().lower() in ("nan", "inconnue", ""):
            continue
        n = len(grp)
        if n < MIN_N:
            continue
        dim_sc = dict(compute_dim_scores(grp, col_map))
        global_sc = (round(sum(dim_sc.values()) / len(dim_sc), 2)
                     if dim_sc else None)
        # bien-être = moyenne bonheur + QdV
        be_vals = []
        for col in [col_bonheur, col_qdv]:
            v = _mean_numeric_col(grp, col)
            if v is not None:
                be_vals.append(v)
        be_sc = round(sum(be_vals) / len(be_vals), 2) if be_vals else None
        conf_sc = _mean_numeric_col(grp, col_confiance)

        commune_scores[commune] = {
            "n": n,
            "global": global_sc,
            "bien_etre": be_sc,
            "confiance": conf_sc,
            "dims": dim_sc,
        }

    docs: List[Tuple[str, str, dict]] = []

    def _sorted(key: str) -> List[Tuple[str, float, int]]:
        return sorted(
            [(c, d[key], d["n"]) for c, d in commune_scores.items() if d.get(key) is not None],
            key=lambda x: -x[1],
        )

    # 1. Global (toutes dimensions)
    rk = _sorted("global")
    if rk:
        text = format_commune_ranking(
            rk, "score moyen toutes dimensions Likert (14 dimensions)",
            note=("Moyenne des 14 dimensions : Transports, Santé, Éducation, Logement, "
                  "Revenus, Emploi, Sécurité, Culture, Services de proximité, Réseau, "
                  "Ratio vie pro/perso, Communauté et relations, Tourisme, Institutions."),
        )
        docs.append(("classement_communes_global", text,
                      {"source_type": "classement_communes", "scope": "communes_global", "criterion": "global_likert"}))
        print(f"  [communes_global] {len(rk)} communes classées")

    # 2. Bien-être (bonheur + QdV)
    rk = _sorted("bien_etre")
    if rk:
        text = format_commune_ranking(
            rk, "bien-être ressenti (bonheur + qualité de vie, échelle 1-5)",
            note="Moyenne des réponses aux questions sur le bonheur général et la qualité de vie globale.",
        )
        docs.append(("classement_communes_bien_etre", text,
                      {"source_type": "classement_communes", "scope": "communes_bien_etre", "criterion": "bien_etre"}))
        print(f"  [communes_bien_etre] {len(rk)} communes classées")

    # 3. Confiance en l'avenir
    rk = _sorted("confiance")
    if rk:
        text = format_commune_ranking(
            rk, "confiance en l'avenir (échelle 1-5)",
        )
        docs.append(("classement_communes_confiance_avenir", text,
                      {"source_type": "classement_communes", "scope": "communes_confiance", "criterion": "confiance_avenir"}))
        print(f"  [communes_confiance] {len(rk)} communes classées")

    # 4. Par dimension
    n_dim_docs = 0
    for dim in col_map.keys():
        ranked_dim = sorted(
            [(c, d["dims"][dim], d["n"]) for c, d in commune_scores.items() if d["dims"].get(dim) is not None],
            key=lambda x: -x[1],
        )
        if len(ranked_dim) < 2:
            continue
        dim_slug = (_norm(dim)
                    .replace(" ", "_").replace("/", "_")
                    .replace("'", "").replace(",", ""))
        text = format_commune_ranking(ranked_dim, f"dimension « {dim} » (Likert 1-5)")
        doc_id = f"classement_communes_dim_{dim_slug}"
        docs.append((doc_id, text,
                      {"source_type": "classement_communes", "scope": "communes_dimension",
                       "criterion": dim, "dimension": dim}))
        n_dim_docs += 1
    print(f"  [communes_dim] {n_dim_docs} classements par dimension")

    return docs


# ── Lecture du CSV ─────────────────────────────────────────────────────────────

def _age_to_range(age_val) -> str:
    """Calcule la tranche d'âge harmonisée (mêmes buckets que portrait_verbatims)."""
    try:
        age = int(float(age_val))
    except (ValueError, TypeError):
        return "Non spécifié"
    if age < 25:   return "18-24"
    if age < 35:   return "25-34"
    if age < 50:   return "35-49"
    if age < 65:   return "50-64"
    return "65+"


def load_df() -> pd.DataFrame:
    df = pd.read_csv(ENQUETE_CSV, encoding="utf-8-sig")

    # Combiner les deux colonnes de commune
    col_as = _find_col(list(df.columns), "commune")
    col_tz = None
    for col in df.columns:
        cn = _norm(col)
        if "commune" in cn and col != col_as:
            col_tz = col
            break
    if col_as and col_tz:
        df["commune"] = df[col_as].fillna(df[col_tz])
    elif col_as:
        df["commune"] = df[col_as]

    # Recalculer cat_age depuis l'âge brut pour harmoniser avec portrait_verbatims
    age_col = _find_col(list(df.columns), "age")
    if age_col:
        df["Catégorie age"] = df[age_col].apply(_age_to_range)

    print(f"  CSV chargé : {len(df)} répondants, {df.get('commune', pd.Series()).nunique()} communes")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def cmd_build(dry_run: bool = False):
    print("=== Build classements dimensions ===")

    print("\n[1/4] Chargement CSV...")
    df = load_df()

    print("\n[2/4] Calcul des classements...")
    docs = build_all_docs(df)
    print(f"  [dimensions] {len(docs)} documents générés")
    commune_ranking_docs = build_commune_rankings(df)
    docs.extend(commune_ranking_docs)
    print(f"  Total : {len(docs)} documents générés")

    if dry_run:
        print("\n[DRY-RUN] Aperçu des documents :")
        for doc_id, text, meta in docs[:3]:
            print(f"\n--- {doc_id} ---")
            print(f"  meta: {meta}")
            print(text[:400])
        return

    print("\n[3/4] Chargement modèle d'embedding...")
    model = get_embed_model()

    print("\n[4/4] Upsert dans ChromaDB...")
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(COLLECTION)
    existing_count = col.count()
    print(f"  Collection '{COLLECTION}' : {existing_count} docs existants")

    ids   = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    metas = [d[2] for d in docs]

    print(f"  Embedding {len(texts)} documents...")
    embeddings = embed_batch(model, texts)

    col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    print(f"  [OK] {len(docs)} documents upsertés -> total {col.count()}")


def cmd_stats():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        print(f"Collection '{COLLECTION}' introuvable")
        return

    results = col.get(
        where={"source_type": "classement"},
        include=["metadatas"],
    )
    metas = results.get("metadatas", [])
    print(f"Documents 'classement' dans '{COLLECTION}' : {len(metas)}")
    from collections import Counter
    scopes = Counter(m.get("scope", "?") for m in metas)
    for scope, cnt in sorted(scopes.items()):
        print(f"  {scope}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit les documents de classement des dimensions")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans upsert dans ChromaDB")
    parser.add_argument("--stats",   action="store_true", help="Affiche les stats des classements existants")
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    else:
        cmd_build(dry_run=args.dry_run)
