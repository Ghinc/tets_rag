"""
migrate_add_questionnaire_id.py

Met à jour les métadonnées de portrait_verbatims dans ChromaDB :
ajoute le champ num_questionnaire en faisant une jointure avec enquete_responses
sur (commune/nom, age_exact, genre, profession/csp).

Ne recalcule PAS les embeddings — mise à jour metadata uniquement.
"""
import math
import sys
import chromadb

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHROMA_PATH = "./chroma_portrait"


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 1. Construire le mapping depuis enquete_responses
    eq = client.get_collection("enquete_responses")
    eq_data = eq.get(include=["metadatas"])

    qid_map = {}
    for m in eq_data["metadatas"]:
        try:
            age_f = float(m.get("age", 0))
            age_i = int(age_f) if not math.isnan(age_f) else None
        except (ValueError, TypeError):
            age_i = None
        if age_i is None:
            continue
        key = (m.get("commune", ""), age_i, m.get("genre", ""), m.get("csp", ""))
        qid_map[key] = int(m.get("respondent_id", -1))

    print(f"Mapping questionnaire : {len(qid_map)} entrées")

    # 2. Charger portrait_verbatims
    vb = client.get_collection("portrait_verbatims")
    vb_data = vb.get(include=["metadatas", "documents", "embeddings"])

    ids = vb_data["ids"]
    docs = vb_data["documents"]
    embeddings = vb_data["embeddings"]
    metadatas = vb_data["metadatas"]

    print(f"portrait_verbatims : {len(ids)} chunks à mettre à jour")

    # 3. Ajouter num_questionnaire à chaque métadonnée
    matched = 0
    updated_metas = []
    for m in metadatas:
        age_exact = m.get("age_exact", -1)
        age_i = int(age_exact) if age_exact not in (-1, None) else None
        key = (m.get("nom", ""), age_i, m.get("genre", ""), m.get("profession", ""))
        qid = qid_map.get(key, -1)
        if qid != -1:
            matched += 1
        new_m = dict(m)
        new_m["num_questionnaire"] = qid
        updated_metas.append(new_m)

    print(f"  Matchés : {matched}/{len(ids)} ({100*matched//len(ids)}%)")
    unmatched = [(m.get("nom"), m.get("age_exact"), m.get("genre"), m.get("profession"))
                 for m, nm in zip(metadatas, updated_metas) if nm["num_questionnaire"] == -1]
    unmatched_unique = set(unmatched)
    if unmatched_unique:
        print(f"  Clés non matchées ({len(unmatched_unique)}) :")
        for k in sorted(unmatched_unique)[:10]:
            print(f"    {k}")

    # 4. Upsert avec les nouvelles métadonnées
    print("\nMise à jour ChromaDB (upsert)...")
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        vb.upsert(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=updated_metas[i:i+batch_size],
        )
        print(f"  Batch {i//batch_size + 1} OK ({min(i+batch_size, len(ids))}/{len(ids)})")

    print(f"\nMigration terminée. Vérification sur 3 docs :")
    check = vb.get(include=["metadatas"], limit=3)
    for m in check["metadatas"]:
        print(f"  {m.get('nom')} age={m.get('age_exact')} qid={m.get('num_questionnaire')}")


if __name__ == "__main__":
    main()
