"""
Indexation des scores OppChoVec par commune dans ChromaDB.

Lit le fichier Excel oppchovec_betti_0_10_trie.xlsx, génère un document
textuel riche par commune, l'encode avec BGE-M3 et le stocke dans la
collection "oppchovec_scores" de ChromaDB.

Usage :
    python index_oppchovec_scores.py
    python index_oppchovec_scores.py --xlsx chemin/vers/fichier.xlsx
    python index_oppchovec_scores.py --stats    # affiche les stats de la collection
"""

import os
import sys
import argparse

XLSX_PATH       = "C:/Users/comiti_g/Downloads/oppchovec_betti_0_10_trie.xlsx"
DIMENSIONS_XLSX = "C:/These/Données2/visualisation_OPPCHOVEC_1011/Code/Python/donnees_oppchovec_par_dimension.xlsx"
CHROMA_PATH     = "./chroma_portrait"
COLLECTION      = "oppchovec_scores"
EMBED_MODEL     = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# Forcer le cache local (pas d'accès internet requis)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.abspath("./model_cache"))


# ── Interprétation textuelle des scores ──────────────────────────────────────

def _level(score: float) -> str:
    if score >= 7.5: return "très élevé"
    if score >= 5.0: return "élevé"
    if score >= 3.0: return "moyen"
    if score >= 1.5: return "faible"
    return "très faible"


def build_document(row, ranks: dict, dim_row=None, dim_norms: dict = None) -> str:
    """
    Génère le texte descriptif complet pour une commune.

    Structure hiérarchique de l'indice OppChoVec :
      OppChoVec
      ├── Opp (Opportunités)  ← sous-score 1/3
      │   ├── Opp1 : Éducation
      │   ├── Opp2 : Diversité sociale
      │   ├── Opp3 : Mobilité
      │   └── Opp4 : Accès TIC
      ├── Cho (Choix)         ← sous-score 2/3
      └── Vec (Vécu)          ← sous-score 3/3

    dim_row   : ligne de donnees_oppchovec_par_dimension.xlsx (ou None)
    dim_norms : dict {Opp1_min, Opp1_max, ...} pour normaliser 0-10 (ou None)
    """
    commune = row["Zone"]
    opp = row["Score_Opp_0_10"]
    cho = row["Score_Cho_0_10"]
    vec = row["Score_Vec_0_10"]
    total = row["OppChoVec_0_10"]

    lines = [
        # Brief hiérarchique — aide le LLM à comprendre la structure
        f"NOTE STRUCTURELLE : L'indice OppChoVec est composé de 3 sous-scores principaux :",
        f"  • Opp (Opportunités) = accès aux ressources du territoire",
        f"  • Cho (Choix) = liberté effective des individus",
        f"  • Vec (Vécu) = réalisations concrètes des habitants",
        f"Le score Opp est lui-même décomposé en 4 sous-dimensions :",
        f"  • Opp1 = niveau d'éducation moyen",
        f"  • Opp2 = diversité sociale (indice de Theil CSP)",
        f"  • Opp3 = accès à la mobilité (voiture et/ou transports en commun)",
        f"  • Opp4 = accès aux TIC (Internet haut débit, 4G)",
        f"Ces scores sont normalisés sur 0-10 par rapport aux 360 communes corses.",
        "",
        f"Score OppChoVec de {commune} : {total:.2f}/10 ({_level(total)}).",
        "",
        f"Sous-scores principaux de {commune} :",
        f"  • Opportunités (Opp) : {opp:.2f}/10 ({_level(opp)})  — rang {ranks['opp']}e/360",
        f"  • Choix (Cho)        : {cho:.2f}/10 ({_level(cho)})  — rang {ranks['cho']}e/360",
        f"  • Vécu (Vec)         : {vec:.2f}/10 ({_level(vec)})  — rang {ranks['vec']}e/360",
        f"  • OppChoVec global   : rang {ranks['total']}e/360",
        "",
    ]

    # Sous-dimensions Opp1/Opp2/Opp3/Opp4 (si disponibles)
    if dim_row is not None and dim_norms is not None:
        def _norm(val, vmin, vmax):
            if vmax == vmin:
                return 5.0
            return max(0.0, min(10.0, (val - vmin) / (vmax - vmin) * 10))

        opp1_n = _norm(dim_row["Opp1"], dim_norms["Opp1_min"], dim_norms["Opp1_max"])
        opp2_n = _norm(dim_row["Opp2"], dim_norms["Opp2_min"], dim_norms["Opp2_max"])
        opp3_n = _norm(dim_row["Opp3"], dim_norms["Opp3_min"], dim_norms["Opp3_max"])
        opp4_n = _norm(dim_row["Opp4"], dim_norms["Opp4_min"], dim_norms["Opp4_max"])

        lines += [
            f"Sous-dimensions du score Opportunités (Opp) de {commune} :",
            f"  • Opp1 — Éducation      : {opp1_n:.2f}/10 ({_level(opp1_n)})  "
            f"[valeur brute : {dim_row['Opp1']:.3f}] — rang {ranks.get('opp1', '?')}e/360",
            f"  • Opp2 — Diversité soc. : {opp2_n:.2f}/10 ({_level(opp2_n)})  "
            f"[indice Theil : {dim_row['Opp2']:.4f}] — rang {ranks.get('opp2', '?')}e/360",
            f"  • Opp3 — Mobilité       : {opp3_n:.2f}/10 ({_level(opp3_n)})  "
            f"[indicateur : {dim_row['Opp3']:.1f}/200] — rang {ranks.get('opp3', '?')}e/360",
            f"  • Opp4 — TIC (numérique): {opp4_n:.2f}/10 ({_level(opp4_n)})  "
            f"[couverture : {dim_row['Opp4']:.1f}%] — rang {ranks.get('opp4', '?')}e/360",
            "",
        ]

    # Analyse comparative (Opp/Cho/Vec)
    comparaisons = []
    if opp > cho and opp > vec:
        comparaisons.append(f"{commune} se distingue surtout par ses opportunités (Opp={opp:.2f}/10)")
    elif cho > opp and cho > vec:
        comparaisons.append(f"{commune} se distingue surtout par ses choix (Cho={cho:.2f}/10)")
    elif vec > opp and vec > cho:
        comparaisons.append(f"{commune} se distingue surtout par le vécu de ses habitants (Vec={vec:.2f}/10)")

    weakest = min([("Opportunités", opp), ("Choix", cho), ("Vécu", vec)], key=lambda x: x[1])
    comparaisons.append(f"Le point le plus faible de {commune} est le score {weakest[0]} ({weakest[1]:.2f}/10)")

    if comparaisons:
        lines.append(f"Analyse de {commune} :")
        for c in comparaisons:
            lines.append(f"- {c}.")
        lines.append("")

    # Valeurs brutes
    lines += [
        f"Valeurs brutes (non normalisées) de {commune} :",
        f"- OppChoVec brut : {row['OppChoVec']:.4f}",
        f"- Opp brut : {row['Score_Opp']:.4f}",
        f"- Cho brut : {row['Score_Cho']:.4f}",
        f"- Vec brut : {row['Score_Vec']:.4f}",
    ]

    return "\n".join(lines)


# ── Script principal ──────────────────────────────────────────────────────────

def cmd_build(xlsx_path: str, dimensions_xlsx: str = DIMENSIONS_XLSX):
    import pandas as pd
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Lecture : {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    print(f"  {len(df)} communes chargées")

    # Calcul des rangs (1 = meilleur score)
    df["rank_total"] = df["OppChoVec_0_10"].rank(ascending=False, method="min").astype(int)
    df["rank_opp"]   = df["Score_Opp_0_10"].rank(ascending=False, method="min").astype(int)
    df["rank_cho"]   = df["Score_Cho_0_10"].rank(ascending=False, method="min").astype(int)
    df["rank_vec"]   = df["Score_Vec_0_10"].rank(ascending=False, method="min").astype(int)

    # Chargement des sous-dimensions Opp1/Opp2/Opp3/Opp4
    dim_df = None
    dim_norms = None
    try:
        dim_df = pd.read_excel(dimensions_xlsx)
        dim_df = dim_df.set_index("Zone")
        # Rangs des sous-dimensions (1 = meilleur)
        dim_df["rank_opp1"] = dim_df["Opp1"].rank(ascending=False, method="min").astype(int)
        dim_df["rank_opp2"] = dim_df["Opp2"].rank(ascending=False, method="min").astype(int)
        dim_df["rank_opp3"] = dim_df["Opp3"].rank(ascending=False, method="min").astype(int)
        dim_df["rank_opp4"] = dim_df["Opp4"].rank(ascending=False, method="min").astype(int)
        # Bornes pour normalisation 0-10
        dim_norms = {
            "Opp1_min": dim_df["Opp1"].min(), "Opp1_max": dim_df["Opp1"].max(),
            "Opp2_min": dim_df["Opp2"].min(), "Opp2_max": dim_df["Opp2"].max(),
            "Opp3_min": dim_df["Opp3"].min(), "Opp3_max": dim_df["Opp3"].max(),
            "Opp4_min": dim_df["Opp4"].min(), "Opp4_max": dim_df["Opp4"].max(),
        }
        print(f"  Sous-dimensions Opp1/Opp2/Opp3/Opp4 chargées ({len(dim_df)} communes)")
    except Exception as e:
        print(f"  AVERTISSEMENT : sous-dimensions Opp non disponibles ({e})")

    print(f"Connexion ChromaDB : {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Recréer la collection
    try:
        client.delete_collection(COLLECTION)
        print(f"  Collection '{COLLECTION}' supprimée (rebuild)")
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"Chargement modèle embeddings ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Génération et indexation des documents...")
    ids, docs, embeddings, metadatas = [], [], [], []

    for _, row in df.iterrows():
        commune = str(row["Zone"]).strip()

        # Données sous-dimensions Opp1-4 pour cette commune (si disponibles)
        dim_row = None
        if dim_df is not None and commune in dim_df.index:
            dim_row = dim_df.loc[commune]

        ranks = {
            "total": int(row["rank_total"]),
            "opp":   int(row["rank_opp"]),
            "cho":   int(row["rank_cho"]),
            "vec":   int(row["rank_vec"]),
        }
        if dim_row is not None:
            ranks["opp1"] = int(dim_row["rank_opp1"])
            ranks["opp2"] = int(dim_row["rank_opp2"])
            ranks["opp3"] = int(dim_row["rank_opp3"])
            ranks["opp4"] = int(dim_row["rank_opp4"])

        doc = build_document(row, ranks, dim_row=dim_row, dim_norms=dim_norms)
        emb = model.encode(f"passage: {doc}").tolist()

        ids.append(f"oppchovec_{commune.lower().replace(' ', '_').replace('-', '_')}")
        docs.append(doc)
        embeddings.append(emb)
        meta = {
            "commune":        commune,
            "oppchovec_0_10": float(row["OppChoVec_0_10"]),
            "opp_0_10":       float(row["Score_Opp_0_10"]),
            "cho_0_10":       float(row["Score_Cho_0_10"]),
            "vec_0_10":       float(row["Score_Vec_0_10"]),
            "rank_total":     int(row["rank_total"]),
            "rank_opp":       int(row["rank_opp"]),
            "rank_cho":       int(row["rank_cho"]),
            "rank_vec":       int(row["rank_vec"]),
            "source":         "oppchovec_betti_0_10",
        }
        if dim_row is not None:
            def _norm(val, vmin, vmax):
                return round(max(0.0, min(10.0, (val - vmin) / (vmax - vmin) * 10)), 4) if vmax != vmin else 5.0
            meta["opp1_0_10"] = _norm(dim_row["Opp1"], dim_norms["Opp1_min"], dim_norms["Opp1_max"])
            meta["opp2_0_10"] = _norm(dim_row["Opp2"], dim_norms["Opp2_min"], dim_norms["Opp2_max"])
            meta["opp3_0_10"] = _norm(dim_row["Opp3"], dim_norms["Opp3_min"], dim_norms["Opp3_max"])
            meta["opp4_0_10"] = _norm(dim_row["Opp4"], dim_norms["Opp4_min"], dim_norms["Opp4_max"])
            meta["rank_opp1"] = int(dim_row["rank_opp1"])
            meta["rank_opp2"] = int(dim_row["rank_opp2"])
            meta["rank_opp3"] = int(dim_row["rank_opp3"])
            meta["rank_opp4"] = int(dim_row["rank_opp4"])
        metadatas.append(meta)

        if len(ids) % 50 == 0:
            print(f"  {len(ids)}/{len(df)} communes traitées...")

    # Indexation par batch de 100
    batch = 100
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i:i+batch],
            documents=docs[i:i+batch],
            embeddings=embeddings[i:i+batch],
            metadatas=metadatas[i:i+batch],
        )

    print(f"\nIndexation terminée : {col.count()} communes dans '{COLLECTION}'")

    # ── Document méthodologique (ID fixe) ────────────────────────────────────
    _index_methodology(col, model)

    # ── Agrégats EPCI + Corse ────────────────────────────────────────────────
    cmd_aggregate()


def _index_methodology(col, model):
    """Indexe (ou met à jour) le document méthodologique OppChoVec."""
    METHODOLOGY_ID = "oppchovec_methodology"
    METHODOLOGY_TEXT = """L'indicateur OppChoVec — Indice de bien-être objectif territorial

L'indicateur OppChoVec a été développé par Lise Bourdeau-Lepage (2011) pour mesurer les inégalités socio-spatiales de bien-être objectif à l'échelle communale. Il opérationnalise l'approche des capabilités d'Amartya Sen (1985) qui considère que l'espace n'est pas neutre pour les individus et que leur épanouissement dépend des conditions objectives offertes par leur territoire. Il est appliqué ici aux 360 communes corses (source : Betti).

DIMENSION 1 — OPPORTUNITÉS (Opp)
Les opportunités objectives offertes aux individus par leur territoire (bien-être en tant que liberté).

• Opp1 — Avoir une bonne éducation (poids 0,25)
  Niveau d'éducation moyen de la population, corrigé de la structure par âge. Calculé comme une somme pondérée des niveaux de diplôme (7 niveaux) par classe d'âge.

• Opp2 — Être inséré dans un environnement social diversifié (poids 0,25)
  Indice de Theil mesurant la diversité des catégories socio-professionnelles (CSP), à la fois en journée (personnes travaillant dans la commune) et la nuit (résidents).

• Opp3 — Avoir les moyens de la mobilité minimale (poids 0,25)
  Proportion de la population disposant d'une voiture et/ou ayant accès à un réseau de transport en commun (train, tram, bus, car) avec une capacité théorique >= 30 min de trajet.

• Opp4 — Avoir accès aux TIC (poids 0,25)
  Couverture locale par Internet haut débit (>= 30 Mbit/s) et réseau 4G.

DIMENSION 2 — CHOIX (Cho)
La liberté de choix effective dont disposent les individus sur ce territoire.

• Cho1 — Ne pas être discriminé(e) (poids 0,5)
  Présence ou absence de quartiers cibles de la politique de la ville dans la commune (indicateur pénalisant la concentration de précarité).

• Cho2 — Avoir les moyens d'influencer les décisions publiques (poids 0,5)
  Proportion d'individus ayant le droit de vote (nationalité française, >= 18 ans) parmi la population en âge de travailler (>= 16 ans).

DIMENSION 3 — VÉCU (Vec)
Les réalisations effectives des individus — ce qui a été réellement accompli et vécu.

• Vec1 — Avoir un revenu décent (poids 0,25)
  Revenu fiscal moyen par foyer fiscal dans la commune.

• Vec2 — Avoir un logement décent (poids 0,25, réparti sur 3 sous-indicateurs)
  - Vec2a : Nombre moyen de personnes par pièce (surpeuplement)
  - Vec2b : Proportion de logements avec équipements sanitaires complets (eau chaude, douche/baignoire, chauffage, cuisine avec évier)
  - Vec2c : Proportion de la population vivant dans une habitation individuelle

• Vec3 — Être bien inséré sur le marché du travail (poids 0,25)
  Stabilité des emplois des résidents (5 niveaux : chômeur, emploi aidé, contrat ponctuel, CDD, emploi stable — CDI, fonctionnaire, indépendant).

• Vec4 — Être proche des services (poids 0,25)
  Nombre d'établissements de vie courante accessibles en moins de 20 minutes (santé, éducation, commerces, culture, administration), calculé par algorithme de plus court chemin (Dijkstra) sur réseau routier.

CALCUL DE L'INDICE

Étape 1 — Standardisation (commensurabilité) :
  V(i,j,k) = (Xijk − Xjk,min) / (Xjk,max − Xjk,min)
  Chaque sous-indicateur est ramené entre 0 (minimum observé) et 1 (maximum observé).

Étape 2 — Agrégation par dimension (moyenne pondérée) :
  Dk(i) = Σ pjk × V(i,j,k)
  Les poids pjk sont égaux au sein de chaque dimension.

Étape 3 — Indice global (avec aversion à la pauvreté et complémentarité) :
  OppChoVec(i) = agrégation des 3 dimensions Dk avec :
    - paramètre β = 2,5 (aversion à la pauvreté : pénalise les très faibles scores)
    - paramètre γ = 1,5 (complémentarité : pénalise les déséquilibres entre dimensions)
  Le résultat brut (0-1) est ensuite renormalisé sur 0-10 pour l'application corse.

INTERPRÉTATION DE L'ÉCHELLE 0-10

- 0 = niveau de bien-être minimal parmi les 360 communes corses
- 10 = niveau de bien-être maximal parmi les 360 communes corses
- ~5 = niveau médian corse

Un score élevé signifie que la commune offre à la fois de bonnes opportunités, une réelle liberté de choix et un vécu satisfaisant. Un score faible peut refléter un déficit sur une ou plusieurs dimensions. Les trois scores partiels (Opp, Cho, Vec) permettent de diagnostiquer la nature du déficit : une commune peut avoir de bonnes opportunités (Opp élevé) mais un vécu difficile (Vec faible), signalant un décalage entre l'offre territoriale et les conditions de vie réelles.

Source : Bourdeau-Lepage L. (2011), adapté pour la Corse. Données : oppchovec_betti_0_10."""

    emb = model.encode(f"passage: {METHODOLOGY_TEXT}").tolist()
    col.upsert(
        ids=[METHODOLOGY_ID],
        documents=[METHODOLOGY_TEXT],
        embeddings=[emb],
        metadatas=[{"source": "methodology", "type": "methodology", "commune": ""}],
    )
    print("OK Doc méthodologique OppChoVec indexé (ID: oppchovec_methodology)")


def cmd_aggregate():
    """Calcule les moyennes OppChoVec par EPCI et pour la Corse entière,
    puis upsert les documents agrégés dans la collection oppchovec_scores."""
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"Connexion ChromaDB : {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ── 1. Charger les scores communes ────────────────────────────────────────
    try:
        col_scores = client.get_collection(COLLECTION)
    except Exception:
        print("Collection oppchovec_scores non trouvée. Lancez --build d'abord.")
        return

    all_scores = col_scores.get(include=["metadatas"])
    commune_scores = {
        m["commune"]: m
        for m in all_scores["metadatas"]
        if m.get("source") == "oppchovec_betti_0_10"
    }
    print(f"  {len(commune_scores)} communes avec scores OppChoVec")

    # ── 2. Charger le mapping commune → EPCI depuis communes_geo ─────────────
    try:
        col_geo = client.get_collection("communes_geo")
    except Exception:
        print("Collection communes_geo non trouvée. Lancez index_geo.py d'abord.")
        return

    all_geo = col_geo.get(include=["metadatas", "documents"])
    epci_map = {}      # {commune: epci}
    epci_info = {}     # {epci: {micro_region, territoire, departement}}
    epci_communes = {} # {epci: [communes]}

    for meta in all_geo["metadatas"]:
        commune = meta.get("commune", "")
        epci = meta.get("epci", "")
        if not commune or not epci or epci == "EPCI inconnu":
            continue
        epci_map[commune] = epci
        epci_info[epci] = {
            "micro_region": meta.get("micro_region", ""),
            "territoire": meta.get("territoire", ""),
            "departement": meta.get("departement", ""),
        }
        epci_communes.setdefault(epci, []).append(commune)

    print(f"  {len(epci_map)} communes avec EPCI | {len(epci_communes)} EPCIs")

    # ── 3. Agréger par EPCI ───────────────────────────────────────────────────
    print("\nChargement modèle embeddings...")
    model = SentenceTransformer(EMBED_MODEL)

    from collections import defaultdict
    epci_data = defaultdict(lambda: {"opp": [], "cho": [], "vec": [], "total": [], "communes": []})

    for commune, scores in commune_scores.items():
        epci = epci_map.get(commune)
        if not epci:
            continue
        epci_data[epci]["total"].append(scores["oppchovec_0_10"])
        epci_data[epci]["opp"].append(scores["opp_0_10"])
        epci_data[epci]["cho"].append(scores["cho_0_10"])
        epci_data[epci]["vec"].append(scores["vec_0_10"])
        epci_data[epci]["communes"].append((commune, scores["oppchovec_0_10"]))

    # Agrégation Corse entière
    all_total = [m["oppchovec_0_10"] for m in commune_scores.values()]
    all_opp   = [m["opp_0_10"]       for m in commune_scores.values()]
    all_cho   = [m["cho_0_10"]       for m in commune_scores.values()]
    all_vec   = [m["vec_0_10"]       for m in commune_scores.values()]

    corse_means = {
        "total": sum(all_total) / len(all_total),
        "opp":   sum(all_opp)   / len(all_opp),
        "cho":   sum(all_cho)   / len(all_cho),
        "vec":   sum(all_vec)   / len(all_vec),
    }
    print(f"\nScore moyen Corse : {corse_means['total']:.2f}/10 (Opp={corse_means['opp']:.2f}, Cho={corse_means['cho']:.2f}, Vec={corse_means['vec']:.2f})")

    # Moyennes Opp1/Opp2/Opp3/Opp4 (si disponibles dans les métadonnées des communes)
    opp1_vals = [m["opp1_0_10"] for m in commune_scores.values() if "opp1_0_10" in m]
    opp2_vals = [m["opp2_0_10"] for m in commune_scores.values() if "opp2_0_10" in m]
    opp3_vals = [m["opp3_0_10"] for m in commune_scores.values() if "opp3_0_10" in m]
    opp4_vals = [m["opp4_0_10"] for m in commune_scores.values() if "opp4_0_10" in m]

    # ── 4. Générer les documents et upsert ────────────────────────────────────
    print("\nGénération et indexation des documents agrégés...")
    agg_ids, agg_docs, agg_embs, agg_metas = [], [], [], []

    # Document classement global des communes (liste complète avec EPCI)
    communes_sorted = sorted(commune_scores.items(), key=lambda x: -x[1]["oppchovec_0_10"])
    median_val = communes_sorted[len(communes_sorted) // 2][1]["oppchovec_0_10"]
    ranking_lines = [
        f"Classement OppChoVec des {len(communes_sorted)} communes corses (score 0-10).",
        "Format : rang. Commune [EPCI]  OppChoVec  Opp  Cho  Vec",
        "",
        f"Score moyen Corse : {corse_means['total']:.2f}/10  |  Médiane : {median_val:.2f}/10",
        f"Min : {communes_sorted[-1][0]} ({communes_sorted[-1][1]['oppchovec_0_10']:.2f}/10)  "
        f"Max : {communes_sorted[0][0]} ({communes_sorted[0][1]['oppchovec_0_10']:.2f}/10)",
        "",
    ]
    for rank, (commune, m) in enumerate(communes_sorted, 1):
        epci_label = epci_map.get(commune, "EPCI inconnu")
        ranking_lines.append(
            f"  {rank:3d}. {commune:<28s} [{epci_label:<30s}]"
            f"  {m['oppchovec_0_10']:5.2f}"
            f"  Opp={m['opp_0_10']:5.2f}  Cho={m['cho_0_10']:5.2f}  Vec={m['vec_0_10']:5.2f}"
        )
    ranking_lines += [
        "",
        f"-> COMMUNE EN TETE (rang 1)     : "
        f"{communes_sorted[0][0]} ({communes_sorted[0][1]['oppchovec_0_10']:.2f}/10)",
        f"-> COMMUNE EN DERNIER (rang {len(communes_sorted)}) : "
        f"{communes_sorted[-1][0]} ({communes_sorted[-1][1]['oppchovec_0_10']:.2f}/10)",
    ]
    ranking_text = "\n".join(ranking_lines)
    agg_ids.append("oppchovec_classement_global")
    agg_docs.append(ranking_text)
    agg_embs.append(model.encode(f"passage: {ranking_text}").tolist())
    agg_metas.append({
        "source": "oppchovec_classement",
        "type":   "classement",
        "zone":   "Corse",
        "epci":   "",
    })
    print(f"  [OK] Classement global ({len(communes_sorted)} communes) généré")

    # Document Corse globale
    corse_doc = (
        f"Le score OppChoVec moyen de la Corse (ensemble des 360 communes) est de {corse_means['total']:.2f}/10 ({_level(corse_means['total'])}).\n"
        f"Décomposition : Score Opportunités (Opp) moyen = {corse_means['opp']:.2f}/10, "
        f"Score Choix (Cho) moyen = {corse_means['cho']:.2f}/10, "
        f"Score Vécu (Vec) moyen = {corse_means['vec']:.2f}/10.\n"
        f"Ces moyennes portent sur les {len(all_total)} communes corses disposant d'un score OppChoVec calculé."
    )
    agg_ids.append("oppchovec_aggregate_corse")
    agg_docs.append(corse_doc)
    agg_embs.append(model.encode(f"passage: {corse_doc}").tolist())
    agg_metas.append({
        "source": "oppchovec_aggregate",
        "type": "aggregate",
        "zone": "Corse",
        "epci": "",
        "oppchovec_0_10": round(corse_means["total"], 4),
        "opp_0_10":       round(corse_means["opp"],   4),
        "cho_0_10":       round(corse_means["cho"],   4),
        "vec_0_10":       round(corse_means["vec"],   4),
        "nb_communes":    len(all_total),
    })

    # Document classement des composantes OppChoVec
    main_ranked = sorted(
        [("Opp (Opportunités)", corse_means["opp"]),
         ("Cho (Choix)",        corse_means["cho"]),
         ("Vec (Vécu)",         corse_means["vec"])],
        key=lambda x: -x[1]
    )
    classement_lines = [
        f"Classement des composantes OppChoVec — Corse entière ({len(all_total)} communes).",
        "",
        "Sous-scores principaux (du plus élevé au plus faible) :",
    ]
    for i, (name, score) in enumerate(main_ranked, 1):
        classement_lines.append(f"  {i}. {name:<25s} : {score:.2f}/10 ({_level(score)})")
    classement_lines += [
        "",
        f"-> COMPOSANTE LA PLUS ELEVEE  : {main_ranked[0][0]} ({main_ranked[0][1]:.2f}/10)",
        f"-> COMPOSANTE LA PLUS FAIBLE  : {main_ranked[-1][0]} ({main_ranked[-1][1]:.2f}/10)",
    ]
    if opp1_vals and opp2_vals and opp3_vals and opp4_vals:
        opp_sub = sorted([
            ("Opp1 — Éducation",          sum(opp1_vals) / len(opp1_vals)),
            ("Opp2 — Diversité sociale",   sum(opp2_vals) / len(opp2_vals)),
            ("Opp3 — Mobilité",           sum(opp3_vals) / len(opp3_vals)),
            ("Opp4 — Accès TIC",          sum(opp4_vals) / len(opp4_vals)),
        ], key=lambda x: -x[1])
        classement_lines += [
            "",
            f"Sous-dimensions du score Opportunités (Opp), classées :",
        ]
        for i, (name, score) in enumerate(opp_sub, 1):
            classement_lines.append(f"  {i}. {name:<30s} : {score:.2f}/10 ({_level(score)})")
        classement_lines += [
            "",
            f"-> SOUS-DIMENSION OPP LA PLUS ELEVEE : {opp_sub[0][0]} ({opp_sub[0][1]:.2f}/10)",
            f"-> SOUS-DIMENSION OPP LA PLUS FAIBLE : {opp_sub[-1][0]} ({opp_sub[-1][1]:.2f}/10)",
        ]
    classement_text = "\n".join(classement_lines)
    agg_ids.append("oppchovec_classement_composantes")
    agg_docs.append(classement_text)
    agg_embs.append(model.encode(f"passage: {classement_text}").tolist())
    agg_metas.append({
        "source": "oppchovec_classement",
        "type":   "classement",
        "zone":   "Corse",
        "epci":   "",
    })
    print("  [OK] Document classement composantes OppChoVec généré")

    # Documents par EPCI
    for epci, data in sorted(epci_data.items()):
        n = len(data["total"])
        if n == 0:
            continue
        m_total = sum(data["total"]) / n
        m_opp   = sum(data["opp"])   / n
        m_cho   = sum(data["cho"])   / n
        m_vec   = sum(data["vec"])   / n

        # Top/bottom communes
        communes_sorted = sorted(data["communes"], key=lambda x: -x[1])
        top3    = communes_sorted[:3]
        bottom3 = communes_sorted[-3:]

        info = epci_info.get(epci, {})
        mr   = info.get("micro_region", "") or "micro-région inconnue"
        terr = info.get("territoire", "")   or "territoire inconnu"

        doc = (
            f"Le score OppChoVec moyen de la CC {epci} est de {m_total:.2f}/10 ({_level(m_total)}).\n"
            f"Cette intercommunalité est située dans la micro-région {mr} (territoire : {terr}).\n"
            f"Décomposition : Opportunités (Opp) = {m_opp:.2f}/10, Choix (Cho) = {m_cho:.2f}/10, Vécu (Vec) = {m_vec:.2f}/10.\n"
            f"Elle regroupe {n} communes dont les scores sont disponibles.\n"
            f"Les {min(3,len(top3))} meilleures communes : {', '.join(f'{c} ({s:.2f})' for c,s in top3)}.\n"
            f"Les {min(3,len(bottom3))} communes avec les scores les plus faibles : {', '.join(f'{c} ({s:.2f})' for c,s in bottom3)}.\n"
            f"Toutes les communes de la CC {epci} avec leur score OppChoVec : "
            + ", ".join(f"{c} ({s:.2f})" for c, s in sorted(data["communes"], key=lambda x: x[0])) + "."
        )

        safe_epci = epci.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace("'", '')
        safe_id = f"oppchovec_aggregate_epci_{safe_epci}"
        agg_ids.append(safe_id)
        agg_docs.append(doc)
        agg_embs.append(model.encode(f"passage: {doc}").tolist())
        agg_metas.append({
            "source": "oppchovec_aggregate",
            "type": "aggregate",
            "zone": epci,
            "epci": epci,
            "micro_region": mr,
            "territoire": terr,
            "oppchovec_0_10": round(m_total, 4),
            "opp_0_10":       round(m_opp,   4),
            "cho_0_10":       round(m_cho,   4),
            "vec_0_10":       round(m_vec,   4),
            "nb_communes":    n,
        })
        print(f"  {epci}: {m_total:.2f}/10 ({n} communes)")

    # Upsert dans oppchovec_scores
    col_scores.upsert(
        ids=agg_ids,
        documents=agg_docs,
        embeddings=agg_embs,
        metadatas=agg_metas,
    )
    print(f"\nOK {len(agg_ids)} documents agrégés upsertés (1 Corse + {len(agg_ids)-1} EPCIs)")
    print(f"Total collection : {col_scores.count()} documents")


def cmd_stats():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        print("Collection non trouvée. Lancez le script sans --stats d'abord.")
        return

    all_data = col.get(include=["metadatas"])
    metas = all_data["metadatas"]
    scores = [m["oppchovec_0_10"] for m in metas]

    print(f"\nCollection : {COLLECTION}")
    print(f"Total communes : {col.count()}")
    print(f"OppChoVec moyen : {sum(scores)/len(scores):.2f}/10")
    print(f"OppChoVec min   : {min(scores):.2f}/10 ({[m['commune'] for m in metas if m['oppchovec_0_10'] == min(scores)][0]})")
    print(f"OppChoVec max   : {max(scores):.2f}/10 ({[m['commune'] for m in metas if m['oppchovec_0_10'] == max(scores)][0]})")

    print("\nTop 5 communes :")
    for m in sorted(metas, key=lambda x: -x["oppchovec_0_10"])[:5]:
        print(f"  {m['commune']:30s} {m['oppchovec_0_10']:.2f}/10")
    print("\nBottom 5 communes :")
    for m in sorted(metas, key=lambda x: x["oppchovec_0_10"])[:5]:
        print(f"  {m['commune']:30s} {m['oppchovec_0_10']:.2f}/10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation scores OppChoVec")
    parser.add_argument("--xlsx",      default=XLSX_PATH, help="Chemin vers le fichier Excel")
    parser.add_argument("--stats",     action="store_true", help="Afficher les stats de la collection")
    parser.add_argument("--aggregate", action="store_true", help="(Re)calculer les agrégats EPCI + Corse uniquement")

    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    elif args.aggregate:
        cmd_aggregate()
    else:
        cmd_build(args.xlsx)
