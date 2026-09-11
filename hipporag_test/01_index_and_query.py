"""
HippoRAG v2 — Indexation des 23 fichiers llm_wiki_data + 5 questions de comparaison.
Sources consultables : sol.docs contient le TEXTE COMPLET des passages récupérés.
Sauvegarde : hipporag_results.json (même format que lightrag_results.json / graphrag_results.json).
"""
import sys, os, json, pathlib, time, multiprocessing
multiprocessing.freeze_support()

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR  = pathlib.Path(r"C:\These\llm_wiki_data")
SAVE_DIR  = pathlib.Path(__file__).parent / "hipporag_storage"
OUT       = pathlib.Path(__file__).parent / "hipporag_results.json"

QUESTIONS = [
    ("bien_etre_ajaccio",
     "Quels sont les principaux facteurs de bien-être évoqués par les habitants d'Ajaccio ?"),
    ("corte_vs_bastia",
     "Quelles différences observe-t-on entre Corte et Bastia en matière de revenus et de logement ?"),
    ("etudiants_qov",
     "Comment les étudiants perçoivent-ils leur qualité de vie en Corse ?"),
    ("scores_satisfaction",
     "Quelles communes ont les scores de satisfaction les plus élevés dans l'enquête ?"),
    ("environnement_bienetre",
     "Quels liens existent entre l'environnement naturel et le bien-être subjectif en Corse ?"),
]


MAX_CHARS = 20000  # ≈ 5000 tokens, sous la limite 8192 de text-embedding-3-small
CHUNK_SIZE = 3000  # chars par chunk pour les grands fichiers
CHUNK_OVERLAP = 200


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Découpe un texte en chunks de taille fixe avec overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def load_docs():
    files = (
        list(DATA_DIR.rglob("*.txt")) +
        list(DATA_DIR.rglob("*.md")) +
        list(DATA_DIR.rglob("*.csv"))
    )
    docs = []
    for f in sorted(files):
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 50:
            print(f"  [SKIP] {f.name} (trop court)")
            continue
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2].strip()
        if len(text) <= MAX_CHARS:
            docs.append(f"[Source: {f.name}]\n{text}")
            print(f"  ✓ {f.name}  ({len(text):,} chars → 1 chunk)")
        else:
            chunks = split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(f"[Source: {f.name}]\n{chunk}")
            print(f"  ✓ {f.name}  ({len(text):,} chars → {len(chunks)} chunks)")
    return docs


def main():
    # Imports hipporag ici pour éviter les problèmes multiprocessing.Manager() au niveau module sur Windows
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    SAVE_DIR.mkdir(exist_ok=True)

    config = BaseConfig(
        preprocess_chunk_max_token_size=512,
        preprocess_chunk_overlap_token_size=64,
        qa_top_k=5,
        retrieval_top_k=50,
        force_index_from_scratch=False,
    )

    print("=== HippoRAG v2 — Chargement des documents ===\n")
    docs = load_docs()
    print(f"\n{len(docs)} fichiers chargés.\n")

    print("=== HippoRAG v2 — Indexation + Requêtes ===\n")
    results = []

    hipporag = HippoRAG(
        global_config=config,
        save_dir=str(SAVE_DIR),
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
    )

    # Indexation (skip si index déjà présent)
    print("→ Indexation en cours (peut prendre 5-15 min à la première exécution)…")
    t0 = time.time()
    hipporag.index(docs=docs)
    print(f"  ✓ Indexation terminée ({time.time()-t0:.0f}s)\n")

    # Requêtes
    for qid, question in QUESTIONS:
        print(f"Q [{qid}] : {question}")
        try:
            t0 = time.time()
            query_solutions, _, _ = hipporag.rag_qa(queries=[question])
            elapsed = time.time() - t0
            sol = query_solutions[0]

            answer = sol.answer or ""
            # Sources : passages originaux avec scores PPR
            sources = []
            docs_list   = sol.docs or []
            scores_list = list(sol.doc_scores) if sol.doc_scores is not None else [0.0] * len(docs_list)
            for i, (doc_text, score) in enumerate(zip(docs_list, scores_list)):
                # Extraire le nom de fichier depuis le préfixe "[Source: xxx]"
                file_name = "—"
                clean_text = doc_text
                if doc_text.startswith("[Source:"):
                    first_line, _, rest = doc_text.partition("\n")
                    file_name = first_line.replace("[Source:", "").strip().rstrip("]")
                    clean_text = rest.strip()
                sources.append({
                    "rank": i + 1,
                    "score": float(score),
                    "text": clean_text,
                    "file": file_name,
                })

            print(f"  → {len(answer)} chars réponse | {len(sources)} sources | {elapsed:.0f}s\n")
        except Exception as e:
            print(f"  ERREUR: {e}\n")
            answer = f"[ERREUR: {e}]"
            sources = []

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "sources": sources,
        })

    payload = {
        "system": "HippoRAG v2",
        "version": "2.0.0a3",
        "index_dir": str(SAVE_DIR),
        "files_indexed": len(docs),
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ hipporag_results.json sauvegardé ({len(results)} questions)")
    print("\nLancez maintenant 02_judge_hipporag.py")


if __name__ == "__main__":
    main()
