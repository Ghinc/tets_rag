"""
Ajoute une ligne d'en-tête temporelle au début de chaque document ChromaDB
pour les collections qui n'ont pas d'information d'année.

OppChoVec    → indicateurs 2020-2022, calcul 2025
Enquête      → collecte 2024-2026 (en cours)
Entretiens   → réalisés en 2022
"""

import chromadb, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BATCH = 200  # update en lots

YEAR_PREFIXES = {
    # OppChoVec
    "oppchovec_scores": (
        "[Millésime des données : indicateurs INSEE/statistiques 2020-2022, "
        "score OppChoVec calculé en 2025. "
        "Aucune donnée disponible pour d'autres années (pas de série temporelle).]\n"
    ),
    # Enquête citoyenne
    "enquete_responses":      "[Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    "communes_profil":        "[Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    "enquete_scores_commune": "[Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    "raptor_summaries":       "[Synthèse RAPTOR — Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    "raptor_quanti_summaries":"[Synthèse RAPTOR quanti — Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    "raptor_enquete_summaries":"[Synthèse RAPTOR — Enquête citoyenne Qualité de vie en Corse — Collecte : 2024-2026 (enquête en cours).]\n",
    # Entretiens
    "portrait_entretiens":          "[Entretiens semi-directifs — Réalisés en 2022.]\n",
    "portrait_verbatims":           "[Entretiens semi-directifs — Réalisés en 2022.]\n",
    "raptor_entretiens_summaries":  "[Synthèse RAPTOR — Entretiens semi-directifs réalisés en 2022.]\n",
}


def update_collection(client, col_name, prefix):
    col = client.get_collection(col_name)
    total = col.count()
    print(f"\n  {col_name} ({total} docs)")

    updated = 0
    skipped = 0
    offset = 0

    while offset < total:
        batch = col.get(limit=BATCH, offset=offset, include=["documents", "metadatas", "embeddings"])
        ids   = batch["ids"]
        docs  = batch["documents"]
        metas = batch["metadatas"]

        new_ids, new_docs, new_metas, new_embs = [], [], [], []
        for doc_id, doc, meta, emb in zip(ids, docs, metas, batch["embeddings"]):
            if doc.startswith(prefix):
                skipped += 1
            else:
                new_ids.append(doc_id)
                new_docs.append(prefix + doc)
                new_metas.append(meta)
                new_embs.append(emb)

        if new_ids:
            col.update(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metas,
                embeddings=new_embs,
            )
            updated += len(new_ids)

        offset += len(ids)
        if len(ids) < BATCH:
            break

    print(f"    → {updated} mis à jour, {skipped} déjà OK")


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="chroma_portrait")

    print("=== Mise à jour des années dans ChromaDB ===")
    for col_name, prefix in YEAR_PREFIXES.items():
        update_collection(client, col_name, prefix)

    print("\n=== Terminé ===")
