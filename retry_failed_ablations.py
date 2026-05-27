"""Retry the 4 rate-limited questions in the ablation JSON."""
import json, requests, time

JSON_FILE = "comparaisons_rag/ablations_20q_20260513_101927.json"
BASE = "http://localhost:8000/api/query"
HEADERS = {"Content-Type": "application/json"}

with open(JSON_FILE, encoding="utf-8") as f:
    data = json.load(f)

failed = [
    ("v_decomp",        68, "Quel est le score moyen, euh, moyen, tu vois, du bien-être général ?",                                        5),
    ("v_decomp",        76, "Un faible bien-être perçu peut-il influencer la manière dont les habitants évaluent leur territoire ?",        5),
    ("v_decomp_raptor",  9, "Combien d'habitants ont répondu à l'enquête à Ajaccio ?",                                                      5),
    ("v_decomp_raptor", 52, "Les données disponibles permettent-elles d'analyser spécifiquement le bien-être des retraités ?",              5),
]

print("Retry des 4 questions (rate-limit 429) avec délai 30s entre chaque...", flush=True)
for ver, row, question, k in failed:
    print(f"\n[{ver}] R{row}: {question[:60]}...", flush=True)
    print("  Attente 30s...", flush=True)
    time.sleep(30)
    try:
        t0 = time.time()
        r = requests.post(BASE, json={"question": question, "rag_version": ver, "k": k},
                          headers=HEADERS, timeout=300)
        elapsed = time.time() - t0
        if r.status_code == 200:
            d = r.json()
            idx = next(i for i, e in enumerate(data[ver]) if e.get("excel_row") == row)
            data[ver][idx].update({
                "answer": d.get("answer", ""),
                "n_sources": len(d.get("sources", [])),
                "n_subquestions": len(d.get("sub_questions") or []),
                "elapsed_s": round(elapsed, 1),
                "status": "ok",
                "error": None,
            })
            print(f"  OK | {elapsed:.1f}s | {data[ver][idx]['n_sources']}src", flush=True)
        else:
            print(f"  ERREUR {r.status_code}: {r.text[:150]}", flush=True)
    except Exception as e:
        print(f"  EXCEPTION: {e}", flush=True)

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n--- RESUME FINAL ---", flush=True)
for ver, entries in data.items():
    ok = sum(1 for e in entries if e.get("status") == "ok")
    err = 20 - ok
    print(f"  {ver:<20} : {ok}/20 OK, {err} err", flush=True)
