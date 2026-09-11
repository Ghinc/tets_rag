"""
GraphRAG — Phase 2 : pose les 5 questions de comparaison.
Mode 'global' pour les questions d'agrégation, 'local' pour les questions ciblées.
Sauvegarde graphrag_results.json.
"""
import subprocess, sys, pathlib, os, json, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GR_DIR = pathlib.Path(__file__).parent

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

PY = sys.executable

QUESTIONS = [
    {
        "id": "bien_etre_ajaccio",
        "question": "Quels sont les principaux facteurs de bien-être évoqués par les habitants d'Ajaccio ?",
        "mode": "local",   # question ciblée sur une entité (Ajaccio)
    },
    {
        "id": "corte_vs_bastia",
        "question": "Quelles différences observe-t-on entre Corte et Bastia en matière de revenus et de logement ?",
        "mode": "local",   # comparaison de deux entités
    },
    {
        "id": "etudiants_qov",
        "question": "Comment les étudiants perçoivent-ils leur qualité de vie en Corse ?",
        "mode": "local",   # groupe démographique ciblé
    },
    {
        "id": "scores_satisfaction",
        "question": "Quelles communes ont les scores de satisfaction les plus élevés dans l'enquête ?",
        "mode": "global",  # question globale / agrégation
    },
    {
        "id": "environnement_bienetre",
        "question": "Quels liens existent entre l'environnement naturel et le bien-être subjectif en Corse ?",
        "mode": "global",  # question transversale / thématique
    },
]

OUT = GR_DIR / "graphrag_results.json"


def query_graphrag(question: str, mode: str) -> str:
    """Interroge GraphRAG via CLI et extrait la réponse."""
    cmd = [
        PY, "-m", "graphrag", "query",
        "--root", str(GR_DIR),
        "--method", mode,
        "--query", question,
        "--response-type", "Multiple Paragraphs",
    ]
    print(f"  $ graphrag query --method {mode} ...")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (r.stdout + "\n" + r.stderr).strip()

    # GraphRAG 2.x imprime le résultat après une ligne contenant "SUCCESS" ou la réponse brute.
    # Pattern typique :
    #   INFO ...
    #   SUCCESS: <mode> Search Response:
    #   <réponse multiligne>
    answer = ""
    patterns = [
        r"SUCCESS[:\s]+(?:Local|Global|Drift|Basic)\s+Search\s+Response[:\s]*\n(.+)",
        r"SUCCESS[:\s]+(.+)",
        r"Response[:\s]+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, output, flags=re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()
            break

    if not answer:
        # Fallback : on prend tout le stdout en ignorant les lignes de log (timestamp)
        lines = [l for l in output.splitlines() if not re.match(r"^\d{4}-\d{2}-\d{2}|^INFO|^ERROR|^WARNING|^DEBUG", l.strip())]
        answer = "\n".join(lines).strip()

    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer


def main():
    print("=== GraphRAG — Requêtes (5 questions) ===\n")

    results = []
    for q_meta in QUESTIONS:
        qid = q_meta["id"]
        question = q_meta["question"]
        mode = q_meta["mode"]
        print(f"[{qid}] {question[:60]}...")
        print(f"  Mode: {mode}")

        try:
            answer = query_graphrag(question, mode)
            print(f"  → {len(answer)} chars\n")
        except Exception as e:
            answer = f"[ERREUR: {e}]"
            print(f"  ERREUR: {e}\n")

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "mode": mode,
            "sources": [],   # GraphRAG ne retourne pas de sources explicites en CLI
        })

    payload = {
        "system": "GraphRAG (Microsoft)",
        "version": "3.x",
        "index_dir": str(GR_DIR),
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ graphrag_results.json sauvegardé ({len(results)} questions)")
    print("\nLancez maintenant 03_judge_graphrag.py")


if __name__ == "__main__":
    main()
