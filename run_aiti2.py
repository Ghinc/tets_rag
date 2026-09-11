import sys, os, time, json
os.environ["LLM_BACKEND"] = "local"
os.environ["PHASE1_STRUCTURED"] = "1"
sys.path.insert(0, r"c:\These\Données2\fichiers_pour_rag")
os.chdir(r"c:\These\Données2\fichiers_pour_rag")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

from local.rag_v12_local import RaptorSubQuestionPipelineV12

rag = RaptorSubQuestionPipelineV12()
rag.init()
t0 = time.time()
ans, *_, metrics = rag.query("Quel est le niveau de bien-être subjectif à Aïti ?")

print("\n--- RÉPONSE ---")
print(ans)
print("--- FIN ---")
print(f"\n[DONE] elapsed={time.time()-t0:.1f}s")

steps = {
    k: {
        "elapsed_s": round(v["elapsed_s"], 1),
        "c_tok":     v["completion_tokens"],
        "tok_s":     round(v["completion_tokens"] / max(v["elapsed_s"], 0.001), 2),
    }
    for k, v in metrics["steps"].items()
}
print(json.dumps(steps, indent=2))
print(f"guard: {json.dumps(metrics.get('guard', {}), ensure_ascii=False)}")
