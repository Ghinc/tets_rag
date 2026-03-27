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

XLSX_PATH    = "C:/Users/comiti_g/Downloads/oppchovec_betti_0_10_trie.xlsx"
CHROMA_PATH  = "./chroma_portrait"
COLLECTION   = "oppchovec_scores"
EMBED_MODEL  = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

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


def build_document(row, ranks: dict) -> str:
    """Génère le texte descriptif complet pour une commune."""
    commune = row["Zone"]
    opp = row["Score_Opp_0_10"]
    cho = row["Score_Cho_0_10"]
    vec = row["Score_Vec_0_10"]
    total = row["OppChoVec_0_10"]

    lines = [
        f"Le score OppChoVec de {commune} est de {total:.2f}/10 ({_level(total)}).",
        "",
        f"Décomposition des sous-scores de {commune} :",
        f"- Score Opportunités (Opp) de {commune} : {opp:.2f}/10 ({_level(opp)})",
        f"- Score Choix (Cho) de {commune} : {cho:.2f}/10 ({_level(cho)})",
        f"- Score Vécu (Vec) de {commune} : {vec:.2f}/10 ({_level(vec)})",
        "",
        f"Rang de {commune} parmi les 360 communes corses :",
        f"- OppChoVec global : {ranks['total']}e sur 360",
        f"- Opportunités (Opp) : {ranks['opp']}e sur 360",
        f"- Choix (Cho) : {ranks['cho']}e sur 360",
        f"- Vécu (Vec) : {ranks['vec']}e sur 360",
        "",
    ]

    # Analyse comparative
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

    # Rappel des valeurs brutes (non normalisées)
    lines += [
        f"Valeurs brutes (non normalisées) de {commune} :",
        f"- OppChoVec brut : {row['OppChoVec']:.4f}",
        f"- Opp brut : {row['Score_Opp']:.4f}",
        f"- Cho brut : {row['Score_Cho']:.4f}",
        f"- Vec brut : {row['Score_Vec']:.4f}",
    ]

    return "\n".join(lines)


# ── Script principal ──────────────────────────────────────────────────────────

def cmd_build(xlsx_path: str):
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
        ranks = {
            "total": int(row["rank_total"]),
            "opp":   int(row["rank_opp"]),
            "cho":   int(row["rank_cho"]),
            "vec":   int(row["rank_vec"]),
        }

        doc = build_document(row, ranks)
        emb = model.encode(f"passage: {doc}").tolist()

        ids.append(f"oppchovec_{commune.lower().replace(' ', '_').replace('-', '_')}")
        docs.append(doc)
        embeddings.append(emb)
        metadatas.append({
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
        })

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
    import chromadb
    from sentence_transformers import SentenceTransformer

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

    # ── 4. Générer les documents et upsert ────────────────────────────────────
    print("\nGénération et indexation des documents agrégés...")
    agg_ids, agg_docs, agg_embs, agg_metas = [], [], [], []

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
