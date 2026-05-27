"""
build_raptor_enquete.py

Construit les synthèses RAPTOR pour deux sources complémentaires :

  1. enquete_responses (246 docs, 70 communes)
     → raptor_enquete_summaries
     Vues : age_range×profession, age_range×commune, profession×commune
            + 1D (age_range, profession, commune, global)
            + dimension×commune, dimension×age_range, dimension×profession, dimension

     Note : les champs dans enquete_responses s'appellent cat_age et csp.
     Le renommage age_range/profession est appliqué à la sortie (métadonnées
     des docs générés) pour aligner avec raptor_summaries (verbatims portrait).

  2. portrait_entretiens (284 chunks, 2 communes : Lozzi, Grossetto-Prugna)
     → raptor_entretiens_summaries
     Vue : commune (seul champ démographique disponible)

Les deux collections de synthèses sont ensuite ajoutées aux _EXTRA_COLLECTIONS
de RaptorRetriever (v9) pour être requêtées en v9, v10 et v11.

Usage :
    python build_raptor_enquete.py --build-enquete      # uniquement enquete_responses
    python build_raptor_enquete.py --build-entretiens   # uniquement portrait_entretiens
    python build_raptor_enquete.py --build-all          # les deux
    python build_raptor_enquete.py --stats              # stats des collections existantes
    python build_raptor_enquete.py --incremental        # ne regenere pas les syntheses existantes
"""

import os
import re
import sys
import json
import time
import argparse
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Forcer UTF-8 sur Windows (évite UnicodeEncodeError dans les print)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Constantes
# ============================================================

CHROMA_PATH = "./chroma_portrait"

# Sources
ENQUETE_SOURCE      = "enquete_responses"
ENTRETIENS_SOURCE   = "portrait_entretiens"

# Cibles
ENQUETE_TARGET      = "raptor_enquete_summaries"
ENTRETIENS_TARGET   = "raptor_entretiens_summaries"

# Seuil de matérialisation (nombre min de docs par groupe)
# Threshold plus bas que verbatims (3) car corpus plus petit
ENQUETE_THRESHOLD   = 2
ENTRETIENS_THRESHOLD = 5  # chunks longs, 5 suffit pour un résumé pertinent

SUMMARIZATION_MODEL    = "mistral-small-latest"
SUMMARIZATION_BASE_URL = "https://api.mistral.ai/v1"

# Renommage des champs source → noms publics dans les métadonnées RAPTOR.
# Les docs enquete_responses utilisent cat_age et csp ; on aligne avec
# raptor_summaries (portrait verbatims) qui utilise age_range et profession.
_FIELD_RENAME = {"cat_age": "age_range", "csp": "profession"}

# Vues pour enquete_responses
# « dimensions » référence les champs dans enquete_responses (cat_age, csp) ;
# les noms de vues et les métadonnées dim*_name utilisent les noms publics
# (age_range, profession) via _FIELD_RENAME.
ENQUETE_VIEWS = [
    # 2D (specificity=2, testées en premier)
    {"name": "enquete_age_range*profession", "dimensions": ["cat_age", "csp"],     "specificity": 2},
    {"name": "enquete_age_range*commune",    "dimensions": ["cat_age", "commune"], "specificity": 2},
    {"name": "enquete_profession*commune",   "dimensions": ["csp",     "commune"], "specificity": 2},
    # 1D (specificity=1, fallback)
    {"name": "enquete_age_range",            "dimensions": ["cat_age"],            "specificity": 1},
    {"name": "enquete_profession",           "dimensions": ["csp"],                "specificity": 1},
    {"name": "enquete_commune",              "dimensions": ["commune"],            "specificity": 1},
    # 0D — baseline Corse entière (specificity=0)
    {"name": "enquete_global",               "dimensions": [],                     "specificity": 0},
]

# Vues dimension QdV × groupe — itèrent sur ENQUETE_DIMENSION_MAP × groupe.
# group_dim : champ dans enquete_responses (cat_age, csp, commune, None).
# Noms de vues et dim*_name dans les métadonnées utilisent les noms publics.
ENQUETE_DIMENSION_VIEWS = [
    {"name": "enquete_dimension*commune",    "group_dim": "commune", "specificity": 2},
    {"name": "enquete_dimension*age_range",  "group_dim": "cat_age", "specificity": 2},
    {"name": "enquete_dimension*profession", "group_dim": "csp",     "specificity": 2},
    {"name": "enquete_dimension",            "group_dim": None,      "specificity": 1},
]

# Seuil min pour les vues dimension (légèrement plus élevé)
ENQUETE_DIMENSION_THRESHOLD = 3

# Mapping dimension QdV → colonnes dans enquete_responses
ENQUETE_DIMENSION_MAP = {
    "Transports":              ["Transports", "TransportsCommun", "ReseauRoutier", "Encombrement"],
    "Santé":                   ["Sante", "MedecinsGeneralistes", "AttenteRDV", "MedecinsSpecialistes"],
    "Éducation":               ["Education"],
    "Logement":                ["Logement"],
    "Revenus":                 ["Revenus"],
    "Emploi":                  ["SituationPro"],
    "Sécurité":                ["Securite"],
    "Culture":                 ["Culture"],
    "Services de proximité":   ["ServicesProximite"],
    "Réseau":                  ["Reseaux"],
    "Ratio vie pro/vie perso": ["EquilibreViePro"],
    "Communauté et relations": ["Entourage", "ImplicationLocale"],
    "Tourisme":                ["Tourisme"],
    "Institutions":            ["Institutions"],
}

# Vues pour portrait_entretiens
ENTRETIENS_VIEWS = [
    {"name": "entretien_commune", "dimensions": ["commune"], "specificity": 1},
]


# ============================================================
# Calcul déterministe des scores Likert
# ============================================================

def _norm(s: str) -> str:
    """Minuscules sans accents (robuste aux ? de re-encodage)."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()

# Mapping label normalisé → score /5
_LIKERT_MAP = {
    # Satisfaction standard
    "tres satisfait":       5, "tres satisfaite":      5,
    "satisfait":            4, "satisfaite":           4,
    "neutre":               3,
    "moyennement satisfait": 3, "moyennement satisfaite": 3,
    "peu satisfait":        2, "peu satisfaite":       2,
    "insatisfait":          1, "insatisfaite":         1,
    "tres peu satisfait":   1, "tres peu satisfaite":  1,
    # Entourage
    "tres bien entoure":    5, "bien entoure":         4,
    "moyennement entoure":  3, "peu entoure":          2,
    "tres peu entoure":     1,
    # Implication
    "tres implique":        5, "implique":             4,
    "moyennement implique": 3, "peu implique":         2,
    "tres peu implique":    1,
}

_ALL_LIKERT_COLS = [col for cols in ENQUETE_DIMENSION_MAP.values() for col in cols]
_GLOBAL_NUM_COLS = [("Bonheur_1_5", "Bonheur"), ("QdV_1_5", "Qualité de vie"), ("Confiance_1_5", "Confiance avenir")]


def _parse_group_stats(chunks: List[Tuple]) -> Dict[str, List[float]]:
    """Extrait les scores numériques de tous les docs du groupe."""
    scores: Dict[str, List[float]] = defaultdict(list)
    kv_pat = re.compile(r'(\w+)=([^,.\n\[]+)')
    for _, text, _ in chunks:
        for key, val in kv_pat.findall(text):
            val = val.strip()
            if key in [c for c, _ in _GLOBAL_NUM_COLS]:
                try:
                    scores[key].append(float(val))
                except ValueError:
                    pass
            elif key in _ALL_LIKERT_COLS:
                n = _LIKERT_MAP.get(_norm(val))
                if n is not None:
                    scores[key].append(float(n))
    return scores


def _format_stats_block(chunks: List[Tuple], dim_filter: List[str] = None) -> str:
    """
    Calcule et formate le bloc de statistiques déterministes.
    dim_filter : si fourni, n'affiche que les dimensions de ENQUETE_DIMENSION_MAP concernées.
    """
    n = len(chunks)
    scores = _parse_group_stats(chunks)
    lines = [f"=== STATISTIQUES CALCULÉES (déterministes, N={n}) ==="]

    # Scores globaux 1-5
    global_parts = []
    for col, label in _GLOBAL_NUM_COLS:
        vals = scores.get(col, [])
        if vals:
            global_parts.append(f"{label} : {sum(vals)/len(vals):.2f}/5 (n={len(vals)})")
    if global_parts:
        lines.append("Scores globaux : " + "  |  ".join(global_parts))

    # Scores par dimension
    lines.append("Scores moyens par dimension (/5) :")
    for dim_label, col_names in ENQUETE_DIMENSION_MAP.items():
        if dim_filter and dim_label not in dim_filter:
            continue
        dim_vals_all = []
        col_parts = []
        for col in col_names:
            vals = scores.get(col, [])
            if vals:
                mean = sum(vals) / len(vals)
                dim_vals_all.append(mean)
                counts = Counter(int(round(v)) for v in vals)
                dist = "  ".join(
                    f"{s}★:{counts.get(s,0)}" for s in [5, 4, 3, 2, 1]
                )
                col_parts.append(f"{col}={mean:.2f} [{dist}]")
        if dim_vals_all:
            dim_mean = sum(dim_vals_all) / len(dim_vals_all)
            detail = "  (" + " / ".join(col_parts) + ")" if col_parts else ""
            lines.append(f"  {dim_label:<28s}: {dim_mean:.2f}/5{detail}")

    lines.append("=== FIN STATISTIQUES ===")
    return "\n".join(lines)


# ============================================================
# LLM
# ============================================================

def _call_llm(prompt: str, system_prompt: str,
              max_tokens: int = 1500, temperature: float = 0.3,
              max_retries: int = 5) -> str:
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY non definie")
    client = OpenAI(api_key=api_key, base_url=SUMMARIZATION_BASE_URL)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=SUMMARIZATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"    [RATE LIMIT] Attente {wait}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


# ============================================================
# Prompts — enquete_responses
# ============================================================

SYSTEM_ENQUETE = (
    "Tu es un analyste spécialisé dans l'analyse des enquêtes de qualité de vie. "
    "Tu produis des synthèses analytiques structurées, factuelles et nuancées "
    "à partir de réponses individuelles à un questionnaire de satisfaction. "
    "Tu ne dois JAMAIS inventer d'information. Base-toi uniquement sur les données fournies."
)


def _prompt_enquete_global(chunks: List[Tuple[str, str, Dict]]) -> str:
    stats_block = _format_stats_block(chunks)
    reponses = [f"[Répondant {i}] {text[:400]}" for i, (_, text, _) in enumerate(chunks[:80], 1)]
    return f"""Voici une synthèse de {len(chunks)} réponses au questionnaire QdV — Corse entière.

{stats_block}

=== ÉCHANTILLON DE RÉPONSES (80 premiers) ===
{chr(10).join(reponses)}

Produis une synthèse analytique globale :

**Corse entière (N={len(chunks)})**

**Satisfaction globale** : Commente les scores moyens (bonheur, qualité de vie, confiance avenir) en citant les valeurs exactes issues des statistiques calculées ci-dessus.

**Toutes les dimensions — tableau complet (classées du score le plus élevé au plus faible)** :
Pour CHAQUE dimension présente dans les statistiques, cite sa moyenne exacte /5 et sa distribution (★5 à ★1). Aucune dimension ne doit être omise.

**Fractures et patterns notables** : Y a-t-il des tensions visibles (ex : jeunes vs seniors, rural vs urbain, sentiments vs conditions objectives) ?

**Limites de l'échantillon** : N={len(chunks)}, biais de représentativité potentiels.

IMPORTANT : les moyennes ci-dessus sont calculées de façon déterministe — utilise-les comme base factuelle, ne les réinvente pas."""


def _prompt_enquete(view_def: Dict, dim_values: Dict[str, str],
                    chunks: List[Tuple[str, str, Dict]]) -> str:
    if view_def["name"] == "enquete_global":
        return _prompt_enquete_global(chunks)

    dims_desc = "\n".join(f"- {d} : {dim_values[d]}" for d in view_def["dimensions"])
    dim_label = ", ".join(f"{d}={dim_values[d]}" for d in view_def["dimensions"])
    stats_block = _format_stats_block(chunks)

    reponses = []
    for i, (_, text, _) in enumerate(chunks, 1):
        reponses.append(f"[Répondant {i}] {text[:600]}")

    return f"""Voici {len(chunks)} réponses au questionnaire qualité de vie correspondant au groupe :
- Vue : {view_def['name']}
{dims_desc}

{stats_block}

=== RÉPONSES BRUTES ===
{chr(10).join(reponses)}

Produis une synthèse analytique de ce groupe en suivant EXACTEMENT ce format.
IMPORTANT : les moyennes ci-dessus sont calculées de façon déterministe — utilise-les comme base factuelle, ne les réinvente pas.

**Groupe : {dim_label} (N={len(chunks)})**

**Satisfaction globale** : Commente les scores globaux (bonheur, qualité de vie, confiance en l'avenir) en t'appuyant sur les moyennes calculées.

**Toutes les dimensions — tableau complet (classées du score le plus élevé au plus faible)** :
Pour CHAQUE dimension présente dans les statistiques, cite sa moyenne exacte /5 et sa distribution (★5 à ★1). Aucune dimension ne doit être omise.

**Patterns notables** : Convergences fortes, contradictions ou spécificités de ce groupe.

**Limites de l'échantillon** : Taille du groupe, biais potentiels.

Reste factuel. Appuie-toi sur les scores calculés."""


# ============================================================
# Prompt — vues dimension QdV × groupe
# ============================================================

SYSTEM_ENQUETE_DIMENSION = (
    "Tu es un analyste spécialisé dans l'analyse des enquêtes de qualité de vie en Corse. "
    "Tu produis des synthèses analytiques focalisées sur une dimension spécifique, "
    "à partir des évaluations individuelles d'un groupe de répondants. "
    "Tu ne dois JAMAIS inventer d'information. Base-toi uniquement sur les données fournies."
)


def _prompt_enquete_dimension(dim_label: str, col_names: List[str],
                               group_desc: str, group_chunks: List[Tuple]) -> str:
    col_list = ", ".join(col_names)
    stats_block = _format_stats_block(group_chunks, dim_filter=[dim_label])

    reponses = []
    for i, (_, text, _) in enumerate(group_chunks, 1):
        reponses.append(f"[Répondant {i}] {text[:600]}")

    return f"""Voici {len(group_chunks)} réponses au questionnaire qualité de vie — {group_desc}.
Dimension à analyser : **{dim_label}**
Colonnes pertinentes : {col_list}

{stats_block}

=== RÉPONSES BRUTES ===
{chr(10).join(reponses)}

Produis une synthèse analytique FOCALISÉE SUR LA DIMENSION "{dim_label}".
IMPORTANT : la distribution et les moyennes ci-dessus sont calculées de façon déterministe — utilise-les, ne les réinvente pas.

**Groupe : {group_desc} (N={len(group_chunks)}) — Dimension : {dim_label}**

**Distribution des évaluations** : Commente la répartition (★5 à ★1) et la moyenne calculée pour chaque colonne [{col_list}].

**Niveau moyen** : Note la moyenne exacte /5 et ce qu'elle traduit qualitativement.

**Patterns notables** : Convergences fortes, sous-groupes ou cas atypiques.

**Comparaison aux autres dimensions** : Cette dimension est-elle mieux ou moins bien évaluée que la moyenne du groupe ?

**Limites** : Taille du groupe, biais possibles.

Reste factuel. Cite les scores exacts."""


# ============================================================
# Prompts — portrait_entretiens
# ============================================================

SYSTEM_ENTRETIENS = (
    "Tu es un analyste spécialisé dans l'analyse d'entretiens semi-directifs sur la qualité de vie. "
    "Tu produis des synthèses analytiques structurées, factuelles et nuancées. "
    "Tu ne dois JAMAIS inventer d'information. Base-toi uniquement sur les extraits fournis."
)


def _prompt_entretiens(view_def: Dict, dim_values: Dict[str, str],
                       chunks: List[Tuple[str, str, Dict]]) -> str:
    dims_desc = "\n".join(f"- {d} : {dim_values[d]}" for d in view_def["dimensions"])
    dim_label = ", ".join(f"{d}={dim_values[d]}" for d in view_def["dimensions"])

    extraits = []
    for i, (_, text, meta) in enumerate(chunks, 1):
        num = meta.get("num_entretien", "?")
        extraits.append(f"[Entretien {num}, extrait {i}] {text[:500]}")

    return f"""Voici {len(chunks)} extraits d'entretiens semi-directifs sur la qualité de vie dans la commune suivante :
- Vue : {view_def['name']}
{dims_desc}

=== EXTRAITS ===
{chr(10).join(extraits)}

Produis une synthèse analytique en suivant EXACTEMENT ce format :

**Commune : {dim_label} — {len(set(c[2].get('num_entretien','?') for c in chunks))} entretiens semi-directifs ({len(chunks)} extraits de transcription)**

**Thèmes dominants** : Quels sujets reviennent le plus souvent dans ces entretiens ?

**Rapport au territoire** : Comment les habitants décrivent-ils leur relation à la commune, au village, à l'espace local ?

**Qualité de vie perçue** : Quels aspects positifs et négatifs de la vie locale ressortent ?

**Signaux faibles** : Y a-t-il des mentions isolées mais significatives (projet, évolution, rupture) ?

**Limites** : Nombre d'entretiens, biais possibles (profil des interviewés, commune représentée).

Reste factuel. Cite des extraits courts entre guillemets quand c'est pertinent."""


# ============================================================
# Builder générique
# ============================================================

def _make_doc_id(view_name: str, dim_values: Dict[str, str]) -> str:
    slug = "_".join(
        v.lower().replace(" ", "-").replace("(", "").replace(")", "")
               .replace("/", "-").replace("'", "")[:30]
        for v in dim_values.values()
    )
    return f"raptor_{view_name}_{slug}"


def build_raptor(
    source_collection_name: str,
    target_collection_name: str,
    views: List[Dict],
    prompt_fn,
    system_prompt: str,
    min_chunks: int,
    incremental: bool = False,
):
    """
    Builder RAPTOR générique — même logique que RaptorBuilder dans rag_v9_raptor.py.
    """
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("=" * 70)
    print(f"RAPTOR : {source_collection_name} -> {target_collection_name}")
    if incremental:
        print("  Mode INCREMENTAL")
    print("=" * 70)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Charger la source
    source = client.get_collection(source_collection_name)
    all_data = source.get(include=["documents", "metadatas"])
    chunks = list(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]))
    print(f"Source : {len(chunks)} chunks")

    # Collection cible
    if incremental:
        target = client.get_or_create_collection(
            target_collection_name, metadata={"hnsw:space": "cosine"}
        )
        existing_ids = set(target.get()["ids"])
        print(f"Cible : {target_collection_name} ({len(existing_ids)} synthèses existantes)")
    else:
        try:
            client.delete_collection(target_collection_name)
        except Exception:
            pass
        client.create_collection(
            target_collection_name, metadata={"hnsw:space": "cosine"}
        )
        target = client.get_collection(target_collection_name)  # re-fetch pour éviter stale ref Windows
        existing_ids = set()
        print(f"Cible : {target_collection_name} (recréée)")

    # Embeddings
    print("Chargement BGE-M3...")
    embed_model = SentenceTransformer("BAAI/bge-m3")

    total = 0
    stats = {}

    for view_def in views:
        print(f"\n--- Vue : {view_def['name']} ---")
        dims = view_def["dimensions"]

        if not dims:
            # Vue globale : un seul groupe = tous les chunks
            valid_groups = {("_global_",): chunks}
            print(f"  Vue globale : {len(chunks)} chunks")
        else:
            # Grouper par dimensions (skip si valeur vide ou 'nan')
            groups = defaultdict(list)
            for chunk_id, text, meta in chunks:
                key_values = tuple(str(meta.get(d, "")).strip() for d in dims)
                if all(v and v.lower() != "nan" for v in key_values):
                    groups[key_values].append((chunk_id, text, meta))

            valid_groups = {k: v for k, v in groups.items() if len(v) >= min_chunks}
            print(f"  {len(groups)} groupes, {len(valid_groups)} avec >= {min_chunks} chunks")

        n_view = 0
        for key_values, group_chunks in valid_groups.items():
            if not dims:
                dim_values = {}
                doc_id = "raptor_enquete_global_corse"
            else:
                dim_values = {d: v for d, v in zip(dims, key_values)}
                doc_id = _make_doc_id(view_def["name"], dim_values)

            if doc_id in existing_ids:
                dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items()) or "global"
                print(f"  [SKIP] {dim_label}")
                n_view += 1
                continue

            try:
                prompt = prompt_fn(view_def, dim_values, group_chunks)
                summary = _call_llm(prompt, system_prompt)
            except Exception as e:
                dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items()) or "global"
                print(f"  [ERREUR] {dim_label}: {e}")
                continue

            embedding = embed_model.encode(f"passage: {summary}").tolist()

            if not dims:
                meta_doc = {
                    "view_name":        "enquete_global",
                    "specificity":      0,
                    "scope":            "corse_entiere",
                    "source_type":      source_collection_name,
                    "dim1_name":        "",
                    "dim1_value":       "",
                    "dim2_name":        "",
                    "dim2_value":       "",
                    "num_chunks":       len(group_chunks),
                    "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                    "built_at":         datetime.now().isoformat(),
                }
            else:
                meta_doc = {
                    "view_name":        view_def["name"],
                    "specificity":      view_def["specificity"],
                    "source_type":      source_collection_name,
                    "dim1_name":        _FIELD_RENAME.get(dims[0], dims[0]),
                    "dim1_value":       key_values[0],
                    "dim2_name":        _FIELD_RENAME.get(dims[1], dims[1]) if len(dims) > 1 else "",
                    "dim2_value":       key_values[1] if len(key_values) > 1 else "",
                    "num_chunks":       len(group_chunks),
                    "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                    "built_at":         datetime.now().isoformat(),
                }

            target.add(
                ids=[doc_id],
                documents=[summary],
                embeddings=[embedding],
                metadatas=[meta_doc],
            )
            n_view += 1
            total += 1

            dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items()) or "Corse entière"
            print(f"  [OK] {dim_label} (N={len(group_chunks)})")
            time.sleep(0.3)

        stats[view_def["name"]] = n_view

    print(f"\n{'=' * 70}")
    print(f"BUILD TERMINÉ : {total} synthèses générées")
    for v, n in stats.items():
        print(f"  {v}: {n}")
    print("=" * 70)
    return stats


# ============================================================
# Document de méthodologie de l'enquête QdV
# ============================================================

def build_enquete_methodology(incremental: bool = False):
    """
    Crée / met à jour le document de description méthodologique de l'enquête QdV
    dans raptor_enquete_summaries (ID fixe : 'enquete_methodology').
    """
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("=" * 70)
    print("MÉTHODOLOGIE ENQUÊTE : construction du document descriptif")
    print("=" * 70)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Carte commune → nombre de répondants
    source = client.get_collection(ENQUETE_SOURCE)
    data = source.get(include=["metadatas"])
    commune_counts = Counter(m.get("commune", "Inconnue") for m in data["metadatas"])
    commune_lines = "\n".join(
        f"  - {c} : {n} répondant{'s' if n > 1 else ''}"
        for c, n in sorted(commune_counts.items(), key=lambda x: -x[1])
        if c and c.lower() not in ("nan", "inconnue", "")
    )
    n_total = len(data["metadatas"])
    n_communes = sum(1 for c in commune_counts if c and c.lower() not in ("nan", "inconnue", ""))

    text = f"""Enquête Qualité de Vie en Corse — Description méthodologique

PRÉSENTATION GÉNÉRALE
Enquête en ligne menée entre 2024 et aujourd'hui auprès de {n_total} habitant·e·s de Corse.
Objectif : recueillir les perceptions subjectives de la qualité de vie (QdV) et du bien-être,
en complément des indicateurs objectifs territoriaux (OppChoVec).
Portée géographique : {n_communes} communes représentées.

RÉPONDANTS PAR COMMUNE (N={n_total})
{commune_lines}

STRUCTURE DU QUESTIONNAIRE (3 volets)

Volet 1 — Dimensions prioritaires (qualitatif)
  Le répondant choisit les 3 dimensions de bien-être les plus importantes pour lui, parmi :
  Transports, Santé, Éducation, Logement, Revenus, Emploi, Sécurité, Culture,
  Services de proximité, Réseau, Ratio vie pro/vie perso, Communauté et relations,
  Tourisme, Institutions.
  Il explique ensuite pourquoi ces dimensions lui semblent prioritaires (texte libre / verbatim).

Volet 2 — Notation Likert des dimensions (quantitatif, échelle 1-5)
  Chaque répondant note sa satisfaction sur les aspects suivants :
  • Transports : mobilité générale, transports en commun, réseau routier, encombrement
  • Santé : accès général, médecins généralistes, délais de RDV, médecins spécialistes
  • Éducation
  • Logement
  • Revenus
  • Emploi (situation professionnelle)
  • Sécurité
  • Culture
  • Services de proximité
  • Réseau (numérique / téléphonie)
  • Ratio vie professionnelle / vie personnelle
  • Communauté et relations (entourage, implication locale)
  • Tourisme
  • Institutions
  Échelle de réponse : 1 = très insatisfait · 2 = peu satisfait · 3 = neutre
                      4 = satisfait · 5 = très satisfait
  (variantes pour "entourage" : très bien entouré → très peu entouré ;
   "implication" : très impliqué → très peu impliqué)

Volet 3 — Questions globales (quantitatif, échelle 1-5)
  • Bonheur général
  • Qualité de vie globale
  • Confiance en l'avenir

SYNTHÈSES DISPONIBLES (collection raptor_enquete_summaries)
  - Par commune (vue enquete_commune)
  - Par tranche d'âge : 15-29, 30-44, 45-59, 60+ (vue enquete_cat_age)
  - Par CSP : étudiant, salarié employé, cadre, fonctionnaire, sans emploi, retraité,
    indépendant... (vue enquete_csp)
  - Croisements 2D : âge×CSP, âge×commune, CSP×commune
  - Par dimension QdV × commune ou âge (vues enquete_dimension*commune,
    enquete_dimension*cat_age)
  - Baseline Corse entière N={n_total} (doc ID : raptor_enquete_global_corse)
"""

    target = client.get_or_create_collection(ENQUETE_TARGET, metadata={"hnsw:space": "cosine"})

    if incremental:
        existing = target.get(ids=["enquete_methodology"], include=["documents"])
        if existing["documents"]:
            print("  [SKIP] enquete_methodology déjà présent (mode incrémental)")
            return

    print("  Chargement BGE-M3...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    emb = embed_model.encode(f"passage: {text}").tolist()

    meta = {
        "view_name":   "enquete_methodology",
        "source_type": "methodology",
        "scope":       "corse_entiere",
        "specificity": 0,
        "dim1_name":   "", "dim1_value": "",
        "dim2_name":   "", "dim2_value": "",
        "num_chunks":  n_total,
        "built_at":    datetime.now().isoformat(),
    }
    target.upsert(
        ids=["enquete_methodology"],
        documents=[text],
        embeddings=[emb],
        metadatas=[meta],
    )
    print(f"  [OK] enquete_methodology upsert ({n_total} répondants, {n_communes} communes)")
    print("=" * 70)


# ============================================================
# Builder vues dimension QdV
# ============================================================

def build_raptor_dimension_views(incremental: bool = False):
    """
    Construit les synthèses RAPTOR focalisées par dimension QdV × groupe.
    Vues : enquete_dimension*commune, enquete_dimension*cat_age, enquete_dimension (1D)
    """
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("=" * 70)
    print(f"RAPTOR DIMENSION : {ENQUETE_SOURCE} -> {ENQUETE_TARGET}")
    if incremental:
        print("  Mode INCREMENTAL")
    print("=" * 70)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    source = client.get_collection(ENQUETE_SOURCE)
    all_data = source.get(include=["documents", "metadatas"])
    chunks = list(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]))
    print(f"Source : {len(chunks)} répondants")

    existing_ids = set()
    if not incremental:
        try:
            client.delete_collection(ENQUETE_TARGET)
        except Exception:
            pass
        client.create_collection(ENQUETE_TARGET, metadata={"hnsw:space": "cosine"})
        target = client.get_collection(ENQUETE_TARGET)  # re-fetch pour éviter stale ref Windows
        print(f"Cible : {ENQUETE_TARGET} (recréée)")
    else:
        target = client.get_or_create_collection(ENQUETE_TARGET, metadata={"hnsw:space": "cosine"})
        existing_ids = set(target.get()["ids"])
        print(f"Cible : {ENQUETE_TARGET} ({len(existing_ids)} synthèses existantes)")

    print("Chargement BGE-M3...")
    embed_model = SentenceTransformer("BAAI/bge-m3")

    total = 0
    stats = {}

    for view_def in ENQUETE_DIMENSION_VIEWS:
        view_name = view_def["name"]
        group_dim = view_def["group_dim"]
        print(f"\n--- Vue : {view_name} ---")

        # Construire les groupes selon group_dim
        if group_dim is None:
            # Vue 1D : un seul groupe = tous les répondants
            groups = {"_all_": chunks}
        else:
            groups = defaultdict(list)
            for chunk_id, text, meta in chunks:
                val = str(meta.get(group_dim, "")).strip()
                if val and val.lower() != "nan":
                    groups[val].append((chunk_id, text, meta))

        valid_groups = {k: v for k, v in groups.items()
                        if len(v) >= ENQUETE_DIMENSION_THRESHOLD}
        print(f"  {len(groups)} groupes, {len(valid_groups)} avec >= {ENQUETE_DIMENSION_THRESHOLD} répondants")
        print(f"  x {len(ENQUETE_DIMENSION_MAP)} dimensions = ~{len(valid_groups) * len(ENQUETE_DIMENSION_MAP)} synthèses max")

        n_view = 0
        for group_key, group_chunks in valid_groups.items():
            for dim_label, col_names in ENQUETE_DIMENSION_MAP.items():
                # Construire dim_values pour l'ID et les métadonnées
                public_dim = _FIELD_RENAME.get(group_dim, group_dim) if group_dim else None
                if group_dim is None:
                    dim_values = {"dimension": dim_label}
                    group_desc = "tous répondants"
                else:
                    dim_values = {"dimension": dim_label, public_dim: group_key}
                    group_desc = f"{public_dim}={group_key}"

                doc_id = _make_doc_id(view_name, dim_values)

                if doc_id in existing_ids:
                    print(f"  [SKIP] {dim_label} / {group_key}")
                    n_view += 1
                    continue

                try:
                    prompt = _prompt_enquete_dimension(
                        dim_label, col_names, group_desc, group_chunks
                    )
                    summary = _call_llm(prompt, SYSTEM_ENQUETE_DIMENSION)
                except Exception as e:
                    print(f"  [ERREUR] {dim_label} / {group_key}: {e}")
                    continue

                embedding = embed_model.encode(f"passage: {summary}").tolist()

                if group_dim is None:
                    meta_doc = {
                        "view_name":        view_name,
                        "specificity":      view_def["specificity"],
                        "source_type":      ENQUETE_SOURCE,
                        "dim1_name":        "dimension",
                        "dim1_value":       dim_label,
                        "dim2_name":        "",
                        "dim2_value":       "",
                        "num_chunks":       len(group_chunks),
                        "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                        "built_at":         datetime.now().isoformat(),
                    }
                else:
                    meta_doc = {
                        "view_name":        view_name,
                        "specificity":      view_def["specificity"],
                        "source_type":      ENQUETE_SOURCE,
                        "dim1_name":        "dimension",
                        "dim1_value":       dim_label,
                        "dim2_name":        public_dim,   # nom public (age_range/profession/commune)
                        "dim2_value":       group_key,
                        "num_chunks":       len(group_chunks),
                        "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                        "built_at":         datetime.now().isoformat(),
                    }

                target.add(
                    ids=[doc_id],
                    documents=[summary],
                    embeddings=[embedding],
                    metadatas=[meta_doc],
                )
                n_view += 1
                total += 1
                print(f"  [OK] {dim_label} / {group_key} (N={len(group_chunks)})")
                time.sleep(0.3)

        stats[view_name] = n_view

    print(f"\n{'=' * 70}")
    print(f"BUILD TERMINÉ : {total} nouvelles synthèses dimension générées")
    for v, n in stats.items():
        print(f"  {v}: {n}")
    print("=" * 70)
    return stats


# ============================================================
# Stats
# ============================================================

def print_stats():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for col_name in [ENQUETE_TARGET, ENTRETIENS_TARGET]:
        try:
            col = client.get_collection(col_name)
            res = col.get(include=["metadatas"])
            views = {}
            for m in res["metadatas"]:
                v = m.get("view_name", "?")
                views[v] = views.get(v, 0) + 1
            print(f"\n{col_name} — {col.count()} synthèses :")
            for v, n in sorted(views.items()):
                print(f"  {v}: {n}")
        except Exception:
            print(f"\n{col_name} : non trouvée")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-enquete",    action="store_true",
                        help="Rebuild vues démographiques enquete_responses (6 vues existantes)")
    parser.add_argument("--build-dimensions", action="store_true",
                        help="Rebuild vues dimension QdV x groupe (nouvelles vues)")
    parser.add_argument("--build-entretiens", action="store_true")
    parser.add_argument("--build-all",        action="store_true",
                        help="Rebuild tout (enquete démo + dimensions + entretiens)")
    parser.add_argument("--stats",            action="store_true")
    parser.add_argument("--incremental",      action="store_true",
                        help="Ne regénère pas les synthèses déjà présentes")
    args = parser.parse_args()

    if args.stats:
        print_stats()

    if args.build_enquete or args.build_all:
        build_raptor(
            source_collection_name=ENQUETE_SOURCE,
            target_collection_name=ENQUETE_TARGET,
            views=ENQUETE_VIEWS,
            prompt_fn=_prompt_enquete,
            system_prompt=SYSTEM_ENQUETE,
            min_chunks=ENQUETE_THRESHOLD,
            incremental=args.incremental,
        )
        build_enquete_methodology(incremental=args.incremental)

    if args.build_dimensions or args.build_all:
        # Quand build_all, la collection vient d'être créée par build_raptor → incremental obligatoire
        dim_incremental = args.incremental or args.build_all
        build_raptor_dimension_views(incremental=dim_incremental)

    if args.build_entretiens or args.build_all:
        build_raptor(
            source_collection_name=ENTRETIENS_SOURCE,
            target_collection_name=ENTRETIENS_TARGET,
            views=ENTRETIENS_VIEWS,
            prompt_fn=_prompt_entretiens,
            system_prompt=SYSTEM_ENTRETIENS,
            min_chunks=ENTRETIENS_THRESHOLD,
            incremental=args.incremental,
        )

    if not any([args.build_enquete, args.build_dimensions, args.build_entretiens,
                args.build_all, args.stats]):
        parser.print_help()
