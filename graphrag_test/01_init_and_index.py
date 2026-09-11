"""
GraphRAG — Phase 1 : init + indexation du corpus llm_wiki_data (23 fichiers).
Utilise gpt-4o-mini pour minimiser le coût.
À exécuter une seule fois depuis le répertoire parent.
"""
import subprocess, sys, pathlib, os, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GR_DIR = pathlib.Path(__file__).parent  # graphrag_test/

# Charge le .env manuellement
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    sys.exit("OPENAI_API_KEY manquante dans .env")

PY = sys.executable


def run(cmd: list, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd or GR_DIR, capture_output=False, text=True)
    if r.returncode != 0:
        print(f"[ERREUR] code {r.returncode}")
        sys.exit(r.returncode)


def patch_settings():
    """Remplace le modèle par gpt-4o-mini et injecte la clé API."""
    settings_path = GR_DIR / "settings.yml"
    if not settings_path.exists():
        sys.exit("settings.yml non trouvé — l'init a-t-elle réussi ?")

    text = settings_path.read_text(encoding="utf-8")

    # Remplace les modèles chers par gpt-4o-mini
    text = re.sub(r"model:\s*gpt-4o\b(?!-mini)", "model: gpt-4o-mini", text)
    text = re.sub(r"model:\s*gpt-4-turbo", "model: gpt-4o-mini", text)

    # Remplace la clé GraphRAG par notre clé OpenAI
    text = text.replace("${GRAPHRAG_API_KEY}", OPENAI_KEY)
    text = text.replace("GRAPHRAG_API_KEY", OPENAI_KEY)

    # Force l'embedding moins cher
    text = re.sub(r"model:\s*text-embedding-ada-002", "model: text-embedding-3-small", text)

    settings_path.write_text(text, encoding="utf-8")
    print("✓ settings.yml patché (gpt-4o-mini + text-embedding-3-small)")
    print("\n--- settings.yml (extrait modèles) ---")
    for line in text.splitlines():
        if any(k in line for k in ["model:", "api_key", "type:"]):
            print(f"  {line}")
    print("---\n")


def main():
    print("=== GraphRAG — Init ===\n")

    # Step 1 : Init si pas encore fait
    if not (GR_DIR / "settings.yml").exists():
        run([PY, "-m", "graphrag", "init", "--root", str(GR_DIR)])
    else:
        print("settings.yml déjà présent, skip init")

    # Step 2 : Patch les modèles
    patch_settings()

    # Step 3 : Indexation
    print("\n=== GraphRAG — Indexation (peut prendre 5–15 min) ===\n")
    run([PY, "-m", "graphrag", "index", "--root", str(GR_DIR)])

    print("\n✓ Indexation terminée. Lancez 02_query_graphrag.py")


if __name__ == "__main__":
    main()
