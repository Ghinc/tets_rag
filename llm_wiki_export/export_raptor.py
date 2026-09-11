"""
Export d'un échantillon de synthèses RAPTOR depuis ChromaDB (lecture SQLite directe)
vers des fichiers Markdown pour import dans LLM Wiki.
Évite l'API ChromaDB qui crashe sur le backfill HNSW.
"""
import sqlite3, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB   = ROOT / "chroma_portrait" / "chroma.sqlite3"
OUT  = pathlib.Path(__file__).parent / "raptor"
OUT.mkdir(exist_ok=True)

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_collection_id(name: str) -> str:
    row = conn.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
    if not row:
        raise ValueError(f"Collection {name!r} introuvable")
    return row["id"]


def get_metadata_segment(col_id: str) -> str:
    row = conn.execute(
        "SELECT id FROM segments WHERE collection=? AND scope='METADATA'",
        (col_id,)
    ).fetchone()
    return row["id"]


def load_collection(col_name: str) -> list[dict]:
    """Retourne tous les docs d'une collection : [{doc, **meta}, ...]"""
    col_id  = get_collection_id(col_name)
    seg_id  = get_metadata_segment(col_id)

    # Tous les embeddings du segment metadata
    emb_rows = conn.execute(
        "SELECT id, embedding_id FROM embeddings WHERE segment_id=?",
        (seg_id,)
    ).fetchall()

    records = []
    for emb in emb_rows:
        int_id  = emb["id"]
        emb_id  = emb["embedding_id"]
        meta_rows = conn.execute(
            "SELECT key, string_value, int_value, float_value, bool_value "
            "FROM embedding_metadata WHERE id=?",
            (int_id,)
        ).fetchall()
        doc  = ""
        meta = {"_id": emb_id}
        for m in meta_rows:
            val = m["string_value"] or m["int_value"] or m["float_value"]
            if m["key"] == "chroma:document":
                doc = m["string_value"] or ""
            else:
                meta[m["key"]] = val
        if doc:
            records.append({"doc": doc, **meta})
    return records


# ── Cache par collection ──────────────────────────────────────────────────────
_cache: dict[str, list[dict]] = {}

def get_col(name: str) -> list[dict]:
    if name not in _cache:
        recs = load_collection(name)
        _cache[name] = recs
        print(f"  [SQLite] {name}: {len(recs)} docs chargés")
    return _cache[name]


# ── Filtre Python ─────────────────────────────────────────────────────────────
def matches(rec: dict, view: str, **kv) -> bool:
    if rec.get("view_name") != view:
        return False
    for k, v in kv.items():
        if str(rec.get(k, "")) != str(v):
            return False
    return True


# ── Export ────────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in s)[:80]


def export_match(col_name: str, view: str, label: str, min_chars=200, **kv) -> int:
    recs = get_col(col_name)
    found = 0
    for rec in recs:
        if not matches(rec, view, **kv):
            continue
        doc = rec["doc"]
        if len(doc.strip()) < min_chars:
            continue

        # Titre
        parts = [label.replace("_", " ")]
        for f in ["dim1_value", "dim2_value"]:
            v = rec.get(f, "")
            if v: parts.append(str(v))
        title = " — ".join(parts)

        # Frontmatter
        fm = ["---", f'title: "{title}"', f'type: "{label}"', f'view: "{view}"']
        for f in ["dim1_name", "dim1_value", "dim2_name", "dim2_value", "n_persons", "num_chunks"]:
            v = rec.get(f)
            if v is not None and str(v) not in ("", "None"):
                fm.append(f'{f}: "{v}"')
        fm += ["---", ""]

        content = "\n".join(fm) + doc.strip()
        fname = OUT / f"{label}_{safe_name(title)}.md"
        fname.write_text(content, encoding="utf-8")
        found += 1
        print(f"  → {fname.name}  ({len(doc):,} chars)")
        break  # on ne veut qu'un exemplaire par filtre
    if not found:
        print(f"  ∅ aucun doc trouvé : view={view} {kv}")
    return found


# ── Plan d'export ─────────────────────────────────────────────────────────────
print("=== Export RAPTOR → llm_wiki_export/raptor/ ===\n")
total = 0

# 1. Synthèses entretiens niveau commune (5 communes)
communes = ["Ajaccio", "Bastia", "Corte", "Porto-Vecchio", "Grosseto-Prugna"]
print(f"[1] Synthèses entretiens par commune")
for c in communes:
    total += export_match("raptor_summaries", "commune", "RAPTOR_entretiens_commune",
                          dim1_value=c)

# 2. Synthèses entretiens par dimension×commune
print(f"\n[2] Synthèses entretiens par dimension×commune")
for commune in ["Ajaccio", "Corte"]:
    for dim in ["Environnement", "Revenus"]:
        total += export_match("raptor_summaries", "dimension*commune",
                              "RAPTOR_entretiens_dim_commune",
                              dim1_value=dim, dim2_value=commune)

# 3. Synthèses entretiens par profession
print(f"\n[3] Synthèses entretiens par profession")
for prof in ["Étudiant(e)", "Retraité(e)"]:
    total += export_match("raptor_summaries", "profession",
                          "RAPTOR_entretiens_profession",
                          dim2_value=prof)

# 4. Synthèses enquête par commune (4 communes)
print(f"\n[4] Synthèses enquête par commune")
for c in ["Ajaccio", "Bastia", "Corte", "Porto-Vecchio"]:
    total += export_match("raptor_enquete_summaries", "enquete_commune",
                          "RAPTOR_enquete_commune",
                          dim1_value=c)

# 5. Synthèse enquête globale
print(f"\n[5] Synthèse enquête globale Corse")
total += export_match("raptor_enquete_summaries", "enquete_global",
                      "RAPTOR_enquete_global", min_chars=100)

# 6. Synthèses enquête par profession
print(f"\n[6] Synthèses enquête par profession")
for prof in ["Étudiant(e)", "Retraité(e)"]:
    total += export_match("raptor_enquete_summaries", "enquete_profession",
                          "RAPTOR_enquete_profession",
                          dim1_value=prof)

conn.close()
print(f"\n=== Total exporté : {total} fichiers Markdown ===")
