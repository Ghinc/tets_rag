"""
build_communes_profil.py
========================
Construit la collection ChromaDB `communes_profil` :
un document par commune avec la grille démographique des répondants
(N, répartition genre, âge, CSP).

Usage :
    python build_communes_profil.py
"""

import re
import unicodedata
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

CHROMA_PATH      = "./chroma_portrait"
COLLECTION_NAME  = "communes_profil"
CSV_PATH         = "donnees_brutes/sortie_questionnaire_traited.csv"
EMBED_MODEL      = "BAAI/bge-m3"

# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Minuscules sans accents, pour les IDs ChromaDB."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total > 0 else "0%"


def _dist_block(title: str, counts: dict, total: int) -> str:
    """Formate un bloc de distribution (genre, âge ou CSP)."""
    lines = [f"{title} :"]
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        if label and str(label) not in ("nan", "0", ""):
            lines.append(f"  - {label} : {n} ({_pct(n, total)})")
    return "\n".join(lines)


# ── Chargement CSV ──────────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

    # Colonnes commune (split A-S / T-Z)
    col_as = next(c for c in df.columns if "commune" in c.lower() and "A" in c)
    col_tz = next(c for c in df.columns if "commune" in c.lower() and "T" in c)
    df["commune"] = df[col_as].fillna(df[col_tz]).str.strip()

    # CSP
    col_csp = next(c for c in df.columns if "socio" in c.lower())
    df["csp"] = df[col_csp].str.strip()

    # Catégorie d'âge
    col_cat = next(c for c in df.columns if "age" in c.lower() and "cat" in c.lower())
    df["cat_age"] = df[col_cat].astype(str).str.strip().replace("0", "Non spécifié")

    return df[["commune", "genre", "cat_age", "csp"]].dropna(subset=["commune"])


# ── Construction des documents ──────────────────────────────────────────────

def build_documents(df: pd.DataFrame) -> list:
    docs = []
    for commune, grp in df.groupby("commune"):
        n = len(grp)

        genre_counts  = grp["genre"].value_counts().to_dict()
        age_counts    = grp["cat_age"].value_counts().to_dict()
        csp_counts    = grp["csp"].value_counts().to_dict()

        text = (
            f"Profil des répondants de la commune de {commune} "
            f"— enquête qualité de vie en Corse\n\n"
            f"Nombre de répondants : {n}\n\n"
            + _dist_block("Répartition par genre", genre_counts, n)
            + "\n\n"
            + _dist_block("Répartition par tranche d'âge", age_counts, n)
            + "\n\n"
            + _dist_block("Répartition par catégorie socioprofessionnelle", csp_counts, n)
        )

        meta = {
            "commune":      commune,
            "n_repondants": n,
            "source_type":  "communes_profil",
            "data_type":    "quanti",
        }

        doc_id = f"profil_{_normalize(commune)}"
        docs.append({"id": doc_id, "text": text, "meta": meta})

    return docs


# ── Indexation ChromaDB ─────────────────────────────────────────────────────

def main():
    print(f"Chargement du CSV ({CSV_PATH})...")
    df = load_dataframe()
    print(f"  {len(df)} répondants, {df['commune'].nunique()} communes")

    docs = build_documents(df)
    print(f"  {len(docs)} documents générés")

    print(f"\nChargement du modèle d'embeddings ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [d["text"] for d in docs]
    print("Calcul des embeddings...")
    embeddings = model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=16, show_progress_bar=True
    ).tolist()

    print(f"\nConnexion ChromaDB ({CHROMA_PATH})...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Recréer proprement la collection
    try:
        existing = client.get_collection(COLLECTION_NAME)
        old_ids = existing.get()["ids"]
        if old_ids:
            existing.delete(ids=old_ids)
        col = existing
        print(f"  Collection existante vidée ({len(old_ids)} docs supprimés)")
    except Exception:
        col = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"  Collection '{COLLECTION_NAME}' créée")

    col.upsert(
        ids=[d["id"] for d in docs],
        documents=texts,
        embeddings=embeddings,
        metadatas=[d["meta"] for d in docs],
    )

    print(f"\nOK : {col.count()} documents indexes dans '{COLLECTION_NAME}'")
    print("\nApercu (3 premiers documents) :")
    for d in docs[:3]:
        preview = d["text"][:400].encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- {d['id']} (N={d['meta']['n_repondants']}) ---")
        print(preview)


if __name__ == "__main__":
    main()
