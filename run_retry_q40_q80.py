"""Retry Q40 et Q80 pour v_decomp_raptor uniquement (erreur réseau lors du premier run)."""
import json, re, sys, time, requests
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import eval_from_excel as evmod
evmod.JUDGE_MODEL = "gpt-4o"; evmod.JUDGE_BASE_URL = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"; evmod._openai_client = None
from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

BACKUP    = "comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET_backup_20260717_070832.json"
RECHECK   = sorted(Path("comparaisons_rag").glob("recheck_5q_*.json"))[-1]
BASE_URL  = "http://localhost:8000/api/query"

with open(BACKUP, encoding="utf-8") as f: backup = json.load(f)
with open(RECHECK, encoding="utf-8") as f: data = json.load(f)

orig = {e["excel_row"]: e for e in backup.get("v_decomp_raptor", [])}

TARGET = [40, 80]
for row_target in TARGET:
    q_info = next((e for e in backup["v_vanilla_k10"] if e["excel_row"] == row_target), None)
    if not q_info: continue
    q = q_info["question"]; section = q_info["section"]; subsection = q_info["subsection"]
    print(f"\nQ{row_target}: {q[:60]}...")

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = requests.post(BASE_URL, json={"question": q, "rag_version": "v_decomp_raptor", "k": 5},
                                 headers={"Content-Type": "application/json"}, timeout=300)
            resp.raise_for_status()
            d = resp.json(); rag_elapsed = round(time.time()-t0,1)
            answer = d.get("answer",""); raw_src = d.get("sources",[])
            src_for_judge = [{"content": s.get("content") or s.get("extrait") or "",
                              "metadata": s.get("metadata",{}),"source_type":s.get("source_type",""),"label":s.get("label","")}
                             for s in raw_src]
            print(f"  RAG {rag_elapsed}s → {len(answer)}ch ({len(raw_src)} src)")
            break
        except Exception as e:
            print(f"  tentative {attempt+1} ERREUR: {e}")
            time.sleep(5)
    else:
        print(f"  Q{row_target} échec après 3 tentatives"); continue

    # Juge sans troncature
    sources_text = _build_sources_text(src_for_judge)
    user_prompt = (f"QUESTION : {q}\n\nSECTION : {section}\n\nSOUS-SECTION : {subsection}\n\n"
                   f"TYPE DE RÉPONSE ATTENDUE : reponse_substantielle_attendue\n\n"
                   f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
                   f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"
                   "Évalue cette réponse selon la procédure et le format spécifiés.\n"
                   "Réponds UNIQUEMENT avec le JSON demandé.")
    time.sleep(1)
    t0 = time.time()
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw); j = json.loads(m.group()) if m else {}
        scores = _parse_judge_v43(j); scores["judge_error"] = None
    except Exception as e:
        scores = {"judge_error": str(e), "score_global": None}
    judge_elapsed = round(time.time()-t0,1)
    sg = scores.get("score_global"); orig_sg = orig.get(row_target,{}).get("score_global")
    print(f"  Juge {judge_elapsed}s → nouveau={sg} ancien={orig_sg}")

    entry = {"excel_row": row_target, "question": q, "section": section, "subsection": subsection,
             "answer_new": answer, "answer_len_new": len(answer),
             "answer_len_old": len(orig.get(row_target,{}).get("answer","")),
             "rag_elapsed_s": rag_elapsed, "judge_elapsed_s": judge_elapsed,
             "sources": src_for_judge,
             **{k:v for k,v in scores.items()},
             "orig_score_global": orig_sg,
             "orig_pertinence": orig.get(row_target,{}).get("pertinence"),
             "orig_fondement_factuel": orig.get(row_target,{}).get("fondement_factuel"),
             "orig_nuance_incertitude": orig.get(row_target,{}).get("nuance_incertitude"),
             "orig_coherence_qualiquanti": orig.get(row_target,{}).get("coherence_qualiquanti"),
             "orig_answer_len": len(orig.get(row_target,{}).get("answer",""))}
    data["v_decomp_raptor"].append(entry)

with open(RECHECK, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nMis à jour : {RECHECK}")
