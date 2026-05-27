"""Reconstruit les index HNSW pour toutes les collections ChromaDB chroma_portrait.

ChromaDB 1.x (Rust backend) ne peut pas lire les fichiers HNSW binaires créés
avec ChromaDB 0.x. Ce script récupère tous les documents, les ré-encode avec
BGE-M3, et les upserte avec leurs embeddings pour forcer la reconstruction.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_portrait"
BATCH = 64

print("Chargement BGE-M3 (SentenceTransformer)...")
embed_model = SentenceTransformer("BAAI/bge-m3")
print("OK\n")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_names = [c.name for c in client.list_collections()]

for cname in collection_names:
    col = client.get_collection(cname)
    total = col.count()
    print(f"[{cname}] {total} docs")

    # Test : HNSW déjà fonctionnel ?
    try:
        test_emb = embed_model.encode(["test"], convert_to_numpy=True).tolist()
        col.query(query_embeddings=[test_emb[0]], n_results=1)
        print("  [OK] HNSW fonctionnel, skip\n")
        continue
    except Exception as e:
        print(f"  [!] HNSW manquant ({type(e).__name__}), reconstruction...")

    # Récupérer tous les docs
    all_ids, all_docs, all_metas = [], [], []
    offset = 0
    PAGE = 1000
    while True:
        batch = col.get(limit=PAGE, offset=offset, include=["documents", "metadatas"])
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"] or [""] * len(batch["ids"]))
        all_metas.extend(batch["metadatas"] or [{}] * len(batch["ids"]))
        offset += PAGE
        if len(batch["ids"]) < PAGE:
            break

    print(f"  Récupéré {len(all_ids)} docs, encodage en cours...")

    all_embs = []
    for i in range(0, len(all_docs), BATCH):
        chunk = all_docs[i:i + BATCH]
        embs = embed_model.encode(chunk, batch_size=BATCH, convert_to_numpy=True)
        all_embs.extend(embs.tolist())
        done = min(i + BATCH, len(all_docs))
        print(f"  Encodage {done}/{len(all_docs)}...", end="\r")
    print()

    # Upsert avec embeddings
    for i in range(0, len(all_ids), BATCH):
        col.upsert(
            ids=all_ids[i:i + BATCH],
            embeddings=all_embs[i:i + BATCH],
            documents=all_docs[i:i + BATCH],
            metadatas=all_metas[i:i + BATCH],
        )

    print(f"  [OK] {len(all_ids)} docs upsertes avec embeddings HNSW\n")

print("Reconstruction HNSW terminée pour toutes les collections.")
