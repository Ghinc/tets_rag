"""
index_geo.py — Intégration géographique : adjacence communes + zones EPCI/micro-régions

Sources :
  - communes-corse (1).geojson : 360 polygones individuels de communes
  - corse_comcom.geojson       : 30 polygones EPCI avec micro-région + territoire

Produit 2 collections ChromaDB :
  - communes_geo  : 1 doc/commune — EPCI, micro-région, territoire, communes adjacentes
  - zones_epci    : 1 doc/EPCI — communes membres, EPCIs voisins
"""

import os, sys, json, io
os.environ["HF_HUB_OFFLINE"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import chromadb
from sentence_transformers import SentenceTransformer

# ── Chemins ─────────────────────────────────────────────────────────────────
COMMUNES_GEO = r"C:\Users\comiti_g\Downloads\communes-corse (1).geojson"
COMCOM_GEO   = r"C:\Users\comiti_g\Downloads\corse_comcom.geojson"
CHROMA_PATH  = "./chroma_portrait"
EMBED_PATH   = "./model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# ── 1. Charger les GeoJSON ───────────────────────────────────────────────────
print("Chargement des GeoJSON...")
communes = gpd.read_file(COMMUNES_GEO)   # 360 polygones communes
comcom   = gpd.read_file(COMCOM_GEO)    # 30 polygones EPCI

print(f"  {len(communes)} communes | {len(comcom)} EPCIs")

# Harmoniser le CRS
comcom = comcom.to_crs(communes.crs)

# ── 2. Spatial join : commune → EPCI/micro-région/territoire ─────────────────
print("\nSpatial join communes → EPCIs...")
# Centroïdes des communes pour le point-in-polygon
communes["centroid"] = communes.geometry.centroid
communes_pts = communes.copy()
communes_pts["geometry"] = communes_pts["centroid"]

joined = gpd.sjoin(
    communes_pts[["code","nom","geometry"]],
    comcom[[
        "geometry",
        "communes_corse_epci_v3_epci",
        "communes_corse_microregions_culturelles_v2_micro_region_reference",
        "communes_corse_microregions_culturelles_v2_territoire_traditionnel",
        "communes_corse_microregions_culturelles_v2_departement"
    ]],
    how="left",
    predicate="within"
)

# Renommer pour lisibilité
joined = joined.rename(columns={
    "communes_corse_epci_v3_epci": "epci",
    "communes_corse_microregions_culturelles_v2_micro_region_reference": "micro_region",
    "communes_corse_microregions_culturelles_v2_territoire_traditionnel": "territoire",
    "communes_corse_microregions_culturelles_v2_departement": "departement",
})
joined = joined.drop_duplicates(subset="nom")
print(f"  {joined['epci'].notna().sum()}/{len(communes)} communes assignées à un EPCI")

# ── 3. Communes adjacentes (polygones voisins = frontière partagée) ──────────
print("\nCalcul des communes adjacentes...")
# Remettre les polygones originaux
communes_orig = communes.merge(
    joined[["nom","epci","micro_region","territoire","departement"]],
    on="nom", how="left"
)

adjacence = {}  # {nom_commune: [noms_voisins]}
for idx, row in communes_orig.iterrows():
    nom = row["nom"]
    geom = row["geometry"]
    # Voisins = communes dont le polygone touche (frontière partagée > simple point)
    voisins = communes_orig[
        (communes_orig.index != idx) &
        (communes_orig.geometry.touches(geom) | communes_orig.geometry.intersects(geom.boundary))
    ]["nom"].tolist()
    # Exclure la commune elle-même
    voisins = [v for v in voisins if v != nom]
    adjacence[nom] = sorted(set(voisins))

n_adj = sum(len(v) for v in adjacence.values())
print(f"  {n_adj} relations d'adjacence calculées (moy. {n_adj/len(adjacence):.1f} voisins/commune)")

# ── 4. Communes par EPCI ────────────────────────────────────────────────────
print("\nGroupement des communes par EPCI...")
epci_communes = communes_orig.groupby("epci")["nom"].apply(sorted).to_dict()
for epci, coms in epci_communes.items():
    print(f"  {epci}: {len(coms)} communes")

# ── 5. EPCIs adjacents (polygones EPCI voisins) ─────────────────────────────
print("\nCalcul des EPCIs adjacents...")
epci_adj = {}
for idx, row in comcom.iterrows():
    nom_epci = row["communes_corse_epci_v3_epci"]
    geom = row["geometry"]
    voisins = comcom[
        (comcom.index != idx) &
        (comcom.geometry.touches(geom) | comcom.geometry.intersects(geom.boundary))
    ]["communes_corse_epci_v3_epci"].tolist()
    epci_adj[nom_epci] = sorted(set(voisins))

# ── 6. Génération des documents texte ───────────────────────────────────────
print("\nGénération des documents texte...")

def dept_label(d):
    if d == "2A": return "Corse-du-Sud (2A)"
    if d == "2B": return "Haute-Corse (2B)"
    return d or "Corse"

# Documents communes_geo
commune_docs = []
for _, row in communes_orig.iterrows():
    nom     = row["nom"]
    epci    = row["epci"] or "EPCI inconnu"
    mr      = row["micro_region"] or "micro-région inconnue"
    terr    = row["territoire"] or "territoire inconnu"
    dept    = dept_label(row.get("departement"))
    voisins = adjacence.get(nom, [])
    membres = [c for c in epci_communes.get(epci, []) if c != nom]

    # Texte principal
    lines = [
        f"La commune de {nom} est située en {dept}.",
        f"Elle appartient à la communauté de communes {epci}, dans la micro-région {mr} (territoire traditionnel : {terr}).",
    ]
    if membres:
        lines.append(f"Les autres communes de sa CC ({epci}) sont : {', '.join(membres[:20])}{'...' if len(membres) > 20 else ''}.")
    if voisins:
        lines.append(f"Ses communes limitrophes sont : {', '.join(voisins[:15])}{'...' if len(voisins) > 15 else ''}.")
    epci_voisins = epci_adj.get(epci, [])
    if epci_voisins:
        lines.append(f"La CC {epci} est voisine des intercommunalités suivantes : {', '.join(epci_voisins)}.")

    doc = " ".join(lines)
    meta = {
        "commune": nom,
        "code_insee": row.get("code", ""),
        "epci": epci,
        "micro_region": mr,
        "territoire": terr,
        "departement": row.get("departement", ""),
        "nb_limitrophes": len(voisins),
        "source": "geojson_geo",
        "data_type": "mixte",
    }
    commune_docs.append({"id": f"geo_{nom}", "doc": doc, "meta": meta})

# Documents zones_epci
zone_docs = []
for _, row in comcom.iterrows():
    epci  = row["communes_corse_epci_v3_epci"]
    mr    = row["communes_corse_microregions_culturelles_v2_micro_region_reference"] or "?"
    terr  = row["communes_corse_microregions_culturelles_v2_territoire_traditionnel"] or "?"
    dept  = dept_label(row.get("communes_corse_microregions_culturelles_v2_departement"))
    coms  = epci_communes.get(epci, [])
    voisins_epci = epci_adj.get(epci, [])

    lines = [
        f"{epci} est une intercommunalité de {dept}, dans la micro-région {mr} (territoire traditionnel : {terr}).",
        f"Elle regroupe {len(coms)} communes : {', '.join(coms)}." if coms else "",
        f"Ses intercommunalités voisines sont : {', '.join(voisins_epci)}." if voisins_epci else "",
    ]
    doc = " ".join(l for l in lines if l)
    meta = {
        "epci": epci,
        "micro_region": mr,
        "territoire": terr,
        "departement": row.get("communes_corse_microregions_culturelles_v2_departement", ""),
        "nb_communes": len(coms),
        "source": "geojson_epci",
        "data_type": "mixte",
    }
    zone_docs.append({"id": f"epci_{epci}", "doc": doc, "meta": meta})

print(f"  {len(commune_docs)} docs communes_geo")
print(f"  {len(zone_docs)} docs zones_epci")

# ── 7. Embedding + indexation ChromaDB ──────────────────────────────────────
print("\nChargement du modèle d'embeddings...")
model = SentenceTransformer(EMBED_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Supprimer et recréer les collections
for col_name in ["communes_geo", "zones_epci"]:
    try:
        client.delete_collection(col_name)
        print(f"  Collection {col_name} supprimée (recréation)")
    except Exception:
        pass

col_communes = client.get_or_create_collection("communes_geo")
col_zones    = client.get_or_create_collection("zones_epci")

print("\nIndexation communes_geo...")
for i, d in enumerate(commune_docs):
    emb = model.encode(f"passage: {d['doc']}").tolist()
    col_communes.add(ids=[d["id"]], documents=[d["doc"]], metadatas=[d["meta"]], embeddings=[emb])
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(commune_docs)}...")

print("\nIndexation zones_epci...")
for d in zone_docs:
    emb = model.encode(f"passage: {d['doc']}").tolist()
    col_zones.add(ids=[d["id"]], documents=[d["doc"]], metadatas=[d["meta"]], embeddings=[emb])

print(f"\n✅ Done :")
print(f"  communes_geo : {col_communes.count()} docs")
print(f"  zones_epci   : {col_zones.count()} docs")

# ── 8. Test rapide ──────────────────────────────────────────────────────────
print("\nTest rapide — 'communes du Niolu' :")
q_emb = model.encode("query: Quelles communes composent le Niolu ?").tolist()
res = col_zones.query(query_embeddings=[q_emb], n_results=2, include=["documents","metadatas","distances"])
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  [{dist:.3f}] {meta['epci']}")
    print(f"  {doc[:200]}")
