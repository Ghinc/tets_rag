"""
test_typing_raptor.py — Test ciblé : effet de la règle de typage sur v_decomp_raptor.

12 questions × 2 configs (v_decomp_raptor vs v_decomp_raptor_no_typing).
Pas de juge global GPT-4o V4.3 — deux signaux de cohérence de registre :
  (a) Signal sources  : famille quali/quanti des sources réellement mobilisées
  (b) Signal LLM étroit : GPT-4o-mini — la réponse contient-elle du registre opposé ?

Usage :
    python test_typing_raptor.py          # run complet
    python test_typing_raptor.py --report  # regen HTML depuis JSONs existants
"""

import argparse, json, os, re, sys, time, requests
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

# ── Constantes ──────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:8000/api/query"
OUT_DIR   = Path("comparaisons_rag/test_typing_raptor")
HTML_PATH = OUT_DIR / f"rapport_typing_raptor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
K         = 5
INTER_Q_DELAY = 2.0   # s entre questions
CONFIGS   = ["v_decomp_raptor", "v_decomp_raptor_no_typing"]

# ── Questions cibles ──────────────────────────────────────────────────────────
# orientation : "objective" | "perception" | "vaste"
# registre_attendu : "quanti" | "quali" | "both"
QUESTIONS = [
    # Objectives — registre QUANTITATIF
    dict(row=4,  orientation="objective",  registre="quanti",
         question="Quel est le score moyen de bien-être à Ajaccio ?"),
    dict(row=5,  orientation="objective",  registre="quanti",
         question="Quelle dimension OppChoVec obtient la note la plus faible à Ajaccio ?"),
    dict(row=6,  orientation="objective",  registre="quanti",
         question="Quel est le score OppChoVec d'Ajaccio, par catégorie ?"),
    dict(row=8,  orientation="objective",  registre="quanti",
         question="Combien d'habitants ont répondu à l'enquête à Ajaccio ?"),
    dict(row=15, orientation="objective",  registre="quanti",
         question="De combien de services de proximité dispose la ville d'Ajaccio ?"),
    # Perception — registre QUALITATIF
    dict(row=9,  orientation="perception", registre="quali",
         question="Comment les 18-25 ans ressentent-ils le bien-être ?"),
    dict(row=10, orientation="perception", registre="quali",
         question="Que pensent les entrepreneurs Ajacciens de la qualité de vie ?"),
    dict(row=11, orientation="perception", registre="quali",
         question="Que pensent les seniors du bien-être à Ajaccio ?"),
    dict(row=14, orientation="perception", registre="quali",
         question="Que révèlent les verbatims sur la sécurité à Corte ?"),
    # Vastes — DEUX familles (témoins de contrôle)
    dict(row=2,  orientation="vaste",      registre="both",
         question="Peut-on considérer Ajaccio comme un territoire favorable au bien-être ?"),
    dict(row=25, orientation="vaste",      registre="both",
         question="Les indicateurs objectifs et les perceptions qualitatives convergent-ils ?"),
    dict(row=35, orientation="vaste",      registre="both",
         question="Observe-t-on un écart significatif entre indicateurs objectifs et perceptions à Bastia ?"),
]

# ── Classification sources quali/quanti ───────────────────────────────────────
_QUALI_PATTERNS = re.compile(
    r"verbatim|entretien|interview|qualitatif|perception|ressenti|opinion|"
    r"enquete_citoyenne|raptor_entretien|raptor_enquete|témoignage|citoyen|subjectif",
    re.IGNORECASE
)
_QUANTI_PATTERNS = re.compile(
    r"oppchovec|score|indicateur|statistique|objectif|chiffre|mesure|"
    r"enquete_score|commune_score|egalite|ratio|index|rang",
    re.IGNORECASE
)

def classify_source(src: dict) -> str:
    """Retourne 'quali', 'quanti' ou 'mixed' pour une source."""
    meta = src.get("metadata", {})
    # Champ 'type' (format v10 RAPTOR)
    stype = str(meta.get("type", "") or meta.get("source_type", "") or "")
    label = str(src.get("label", "") or meta.get("label", "") or "")
    text_hint = stype + " " + label
    is_q = bool(_QUALI_PATTERNS.search(text_hint))
    is_qt = bool(_QUANTI_PATTERNS.search(text_hint))
    if is_q and not is_qt:  return "quali"
    if is_qt and not is_q:  return "quanti"
    if is_q and is_qt:      return "mixed"
    # Fallback: inspecter le contenu
    content = src.get("content", "")[:300].lower()
    if any(kw in content for kw in ("score", "indicateur", "oppchovec", "0.", "1.", "2.", "3.", "4.", "5.")):
        return "quanti"
    return "mixed"

def classify_sources_mobilisees(sm_list: list) -> dict:
    """
    Classe les sources mobilisées (depuis le bloc SOURCES_MOBILISEES de la réponse).
    Retourne {quali: n, quanti: n, mixed: n, dominant: 'quali'|'quanti'|'mixed'|'none'}
    """
    counts = {"quali": 0, "quanti": 0, "mixed": 0}
    for sm in sm_list:
        types_str = " ".join(sm.get("types", []))
        is_q  = bool(_QUALI_PATTERNS.search(types_str))
        is_qt = bool(_QUANTI_PATTERNS.search(types_str))
        if is_q and not is_qt:   counts["quali"]  += 1
        elif is_qt and not is_q: counts["quanti"] += 1
        else:                    counts["mixed"]  += 1
    total = sum(counts.values())
    if total == 0:
        return {**counts, "dominant": "none", "total": 0}
    dominant = max(counts, key=lambda k: counts[k])
    return {**counts, "dominant": dominant, "total": total}

def register_from_sources(raw_sources: list, sm_list: list) -> dict:
    """
    Signal (a) : détermine le registre dominant effectivement utilisé.
    Priorise sources_mobilisees (auto-déclaration Mistral) sur raw_sources.
    """
    if sm_list:
        return classify_sources_mobilisees(sm_list)
    # Fallback sur raw_sources
    counts = {"quali": 0, "quanti": 0, "mixed": 0}
    for s in raw_sources:
        c = classify_source(s)
        counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {**counts, "dominant": "none", "total": 0}
    dominant = max(counts, key=lambda k: counts[k])
    return {**counts, "dominant": dominant, "total": total}

def verdict_a(orientation: str, registre: str, register_info: dict) -> str:
    """
    Signal (a) verdict : COHERENT | VIOLATION | INCOMPLET | N/A
    """
    dominant = register_info.get("dominant", "none")
    total    = register_info.get("total", 0)
    if total == 0 or dominant == "none":
        return "N/A"
    if orientation == "vaste":
        # Les questions vastes doivent mobiliser les deux familles
        has_quali  = register_info.get("quali", 0) + register_info.get("mixed", 0) > 0
        has_quanti = register_info.get("quanti", 0) + register_info.get("mixed", 0) > 0
        if has_quali and has_quanti: return "COHERENT"
        return "INCOMPLET"
    # Questions orientées
    if registre == "quanti":
        # Violation si le dominant est quali
        if dominant == "quali": return "VIOLATION"
        return "COHERENT"
    if registre == "quali":
        if dominant == "quanti": return "VIOLATION"
        return "COHERENT"
    return "COHERENT"

# ── Signal (b) : LLM étroit ────────────────────────────────────────────────────
import importlib
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o-mini"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import _call_llm

_SYSTEM_REGISTER_CHECK = (
    "Tu es un vérificateur de cohérence de registre. "
    "Ta seule tâche : identifier si une réponse contient substantiellement du contenu "
    "du registre OPPOSÉ à celui attendu par la question. "
    "Réponds UNIQUEMENT avec ce JSON exact : "
    '{"violation": true/false, "extrait": "<citation courte du passage fautif ou null>", '
    '"explication": "<1 phrase max>"}'
)

def signal_b(orientation: str, registre: str, question: str, answer: str) -> dict:
    """Signal (b) : appel LLM étroit GPT-4o-mini."""
    if orientation == "vaste":
        return {"verdict": "N/A", "violation": None, "extrait": None, "explication": "Question vaste — les deux registres sont légitimes"}

    opposite = "QUALITATIF (perceptions, verbatims, opinions)" if registre == "quanti" else "QUANTITATIF (scores, indicateurs, chiffres)"
    expected = "QUANTITATIF (scores, indicateurs chiffrés)" if registre == "quanti" else "QUALITATIF (verbatims, perceptions, opinions)"

    prompt = (
        f"Question : {question}\n\n"
        f"Orientation attendue : {expected}\n"
        f"Registre à détecter (registre OPPOSÉ, non attendu) : {opposite}\n\n"
        f"Réponse à analyser (500 premiers chars) :\n{answer[:500]}\n\n"
        f"La réponse présente-t-elle de manière SUBSTANTIELLE du contenu du registre opposé ({opposite}) ?\n"
        f"'Substantielle' = paragraphe entier ou affirmation principale, pas une mention incidente.\n"
        f"Réponds UNIQUEMENT avec le JSON demandé."
    )
    t0 = time.time()
    try:
        raw = _call_llm(_SYSTEM_REGISTER_CHECK, prompt, max_tokens=300, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        violation = bool(j.get("violation", False))
        return {
            "verdict":     "VIOLATION" if violation else "COHERENT",
            "violation":   violation,
            "extrait":     j.get("extrait"),
            "explication": j.get("explication", ""),
            "elapsed_s":   round(time.time() - t0, 1),
        }
    except Exception as e:
        return {"verdict": "ERREUR", "violation": None, "extrait": None, "explication": str(e)[:100]}

# ── Appel API ─────────────────────────────────────────────────────────────────
def call_api(question: str, rag_version: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(API_BASE,
                              json={"question": question, "rag_version": rag_version, "k": K},
                              timeout=360)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    [retry {attempt+1}/{retries}] {e} — attente {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)
    return {"error": "echec après retries", "answer": "", "sources": []}

# ── Run principal ──────────────────────────────────────────────────────────────
def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for q_meta in QUESTIONS:
        row         = q_meta["row"]
        question    = q_meta["question"]
        orientation = q_meta["orientation"]
        registre    = q_meta["registre"]
        q_results   = {"row": row, "question": question, "orientation": orientation,
                       "registre": registre, "configs": {}}

        print(f"\nQ{row:02d} [{orientation:10s}] {question[:60]}")

        for cfg in CONFIGS:
            lbl = "TYPAGE" if "no_typing" not in cfg else "NO_TYPING"
            print(f"  [{lbl}] RAG...", end="", flush=True)
            t0 = time.time()
            resp = call_api(question, cfg)
            elapsed = round(time.time() - t0, 1)

            if "error" in resp and not resp.get("answer"):
                print(f" ERREUR: {resp['error'][:60]}")
                q_results["configs"][cfg] = {"error": resp["error"]}
                continue

            answer   = resp.get("answer", "")
            sources  = resp.get("sources", [])
            sub_q    = resp.get("sub_questions") or []
            sources_mob = resp.get("sources_mobilisees") or []
            print(f" OK ({elapsed}s, {len(sources)} sources, {len(sub_q)} SQ)", end="", flush=True)

            # Signal (a)
            reg_info = register_from_sources(sources, sources_mob)
            verd_a   = verdict_a(orientation, registre, reg_info)

            # Signal (b)
            print(f" LLM-b...", end="", flush=True)
            sig_b = signal_b(orientation, registre, question, answer)

            print(f" a={verd_a} b={sig_b['verdict']}")

            entry = {
                "config":          cfg,
                "elapsed_s":       elapsed,
                "answer":          answer,
                "answer_short":    answer[:400],
                "sub_questions":   sub_q,
                "sources_count":   len(sources),
                "sources_mobilisees": sources_mob,
                "register_info":   reg_info,
                "signal_a":        verd_a,
                "signal_b":        sig_b,
            }
            q_results["configs"][cfg] = entry

            # Sauvegarde incrémentale
            fname = OUT_DIR / f"Q{row:03d}_{cfg}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({"row": row, "question": question, "orientation": orientation,
                           "registre": registre, **entry}, f, ensure_ascii=False, indent=2)

            time.sleep(INTER_Q_DELAY)

        all_results.append(q_results)
        time.sleep(INTER_Q_DELAY)

    # Sauvegarde globale
    global_path = OUT_DIR / "test_typing_raptor_results.json"
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés → {global_path}")
    return all_results

# ── Chargement depuis JSONs existants ─────────────────────────────────────────
def load_results() -> list:
    global_path = OUT_DIR / "test_typing_raptor_results.json"
    if global_path.exists():
        with open(global_path, encoding="utf-8") as f:
            return json.load(f)
    # Reconstruire depuis fichiers individuels
    results = []
    for q_meta in QUESTIONS:
        row = q_meta["row"]
        q_results = {**q_meta, "configs": {}}
        for cfg in CONFIGS:
            fname = OUT_DIR / f"Q{row:03d}_{cfg}.json"
            if fname.exists():
                with open(fname, encoding="utf-8") as f:
                    d = json.load(f)
                q_results["configs"][cfg] = d
        results.append(q_results)
    return results

# ── Génération HTML ───────────────────────────────────────────────────────────
def make_html(results: list) -> str:
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── Calcul des stats synthèse ─────────────────────────────────────────────
    oriented   = [r for r in results if r["orientation"] != "vaste"]
    vastes     = [r for r in results if r["orientation"] == "vaste"]

    def count_violations(subset, cfg, signal):
        n = 0
        for r in subset:
            e = r["configs"].get(cfg, {})
            if signal == "a":
                verdict = e.get("signal_a", "")
            else:
                verdict = (e.get("signal_b") or {}).get("verdict", "")
            if verdict == "VIOLATION":
                n += 1
        return n

    # Summary table data
    n_oriented = len(oriented)
    n_vastes   = len(vastes)
    sa_typ_or  = count_violations(oriented, "v_decomp_raptor",           "a")
    sa_nt_or   = count_violations(oriented, "v_decomp_raptor_no_typing", "a")
    sb_typ_or  = count_violations(oriented, "v_decomp_raptor",           "b")
    sb_nt_or   = count_violations(oriented, "v_decomp_raptor_no_typing", "b")
    # Pour les vastes : INCOMPLET = problème, VIOLATION = impossible par définition
    sa_typ_va  = sum(1 for r in vastes if r["configs"].get("v_decomp_raptor",{}).get("signal_a") == "INCOMPLET")
    sa_nt_va   = sum(1 for r in vastes if r["configs"].get("v_decomp_raptor_no_typing",{}).get("signal_a") == "INCOMPLET")

    # ── Tableau texte brut (pour discussion) ─────────────────────────────────
    print("\n" + "="*70)
    print("TABLEAU DE SYNTHÈSE (texte brut)")
    print("="*70)
    print(f"{'':40s} {'TYPAGE':>12} {'NO_TYPING':>12}")
    print(f"  Questions orientées (n={n_oriented})")
    print(f"    Signal (a) — violations de registre     {sa_typ_or:>8}/{n_oriented}    {sa_nt_or:>8}/{n_oriented}")
    print(f"    Signal (b) — violations de registre     {sb_typ_or:>8}/{n_oriented}    {sb_nt_or:>8}/{n_oriented}")
    print(f"  Questions vastes (n={n_vastes}) — INCOMPLET (contrôle)")
    print(f"    Signal (a) — réponses incomplètes       {sa_typ_va:>8}/{n_vastes}     {sa_nt_va:>8}/{n_vastes}")
    print("="*70 + "\n")

    # ── Verdict global ────────────────────────────────────────────────────────
    delta_a = sa_nt_or - sa_typ_or
    delta_b = sb_nt_or - sb_typ_or
    if delta_a >= 3 or delta_b >= 3:
        verdict_txt = "Le typage a un effet mesurable : No_Typing viole nettement plus le registre."
        verdict_col = "#DC2626"
    elif delta_a <= 0 and delta_b <= 0:
        verdict_txt = "RAPTOR neutralise la règle de typage : violations identiques dans les deux configs."
        verdict_col = "#16A34A"
    else:
        verdict_txt = f"Effet marginal (Δ signal_a={delta_a:+d}, Δ signal_b={delta_b:+d}) — pas de conclusion tranchée."
        verdict_col = "#B45309"

    # ── Helpers HTML ──────────────────────────────────────────────────────────
    def verdict_badge(v):
        if not v:  return '<span class="badge na">—</span>'
        colors = {"COHERENT": "good", "VIOLATION": "bad", "INCOMPLET": "warn",
                  "N/A": "na", "ERREUR": "bad"}
        cls = colors.get(v, "na")
        return f'<span class="badge {cls}">{v}</span>'

    def sq_list(sub_q_list):
        if not sub_q_list:
            return '<em class="muted">—</em>'
        items = []
        for sq in sub_q_list:
            q_text = sq.get("question", "")
            a_text = sq.get("answer", "").strip()
            a_block = (
                f'<details class="sq-ans-details"><summary class="sq-ans-toggle">Réponse</summary>'
                f'<div class="sq-ans-body">{a_text}</div></details>'
            ) if a_text else ""
            items.append(
                f'<li><span class="sq-idx">{sq.get("idx","·")}</span>'
                f' <span class="sq-q">{q_text}</span>'
                f'{a_block}</li>'
            )
        return f'<ol class="sq-list">{"".join(items)}</ol>'

    def sm_pills(sm_list):
        if not sm_list:
            return '<em class="muted">—</em>'
        pills = []
        for sm in sm_list:
            types_str = " · ".join(sm.get("types", ["?"]))
            sq_idx = sm.get("sq", "?")
            is_q  = bool(_QUALI_PATTERNS.search(types_str))
            is_qt = bool(_QUANTI_PATTERNS.search(types_str))
            cls = "pill-quali" if is_q and not is_qt else ("pill-quanti" if is_qt and not is_q else "pill-mixed")
            pills.append(f'<span class="{cls}">SQ{sq_idx}: {types_str[:60]}</span>')
        return "<br>".join(pills)

    def reg_info_display(ri):
        if not ri:
            return "—"
        parts = []
        for k in ("quanti", "quali", "mixed"):
            if ri.get(k, 0):
                parts.append(f"{k}:{ri[k]}")
        return f"dominant={ri.get('dominant','?')} ({', '.join(parts)})"

    # ── Détail Q×Q ────────────────────────────────────────────────────────────
    detail_sections = []
    for r in results:
        row         = r["row"]
        question    = r["question"]
        orientation = r["orientation"]
        registre    = r.get("registre", "?")
        orient_lbl  = {"objective": "Objectif — QUANTI", "perception": "Perception — QUALI", "vaste": "Vaste — BOTH"}[orientation]
        is_vaste    = orientation == "vaste"

        typ_e  = r["configs"].get("v_decomp_raptor", {})
        nt_e   = r["configs"].get("v_decomp_raptor_no_typing", {})

        def col_html(entry, cfg_label):
            if not entry or "error" in entry:
                return f'<div class="col-error">ERREUR : {entry.get("error","?") if entry else "manquant"}</div>'
            va  = entry.get("signal_a", "?")
            vb  = entry.get("signal_b", {})
            ri  = entry.get("register_info", {})
            sub = entry.get("sub_questions", [])
            sm  = entry.get("sources_mobilisees", [])
            ans = entry.get("answer", "")
            b_extr = vb.get("extrait") or ""
            b_expl = vb.get("explication") or ""
            is_violation = va == "VIOLATION" or vb.get("verdict") == "VIOLATION"
            border = ' style="border-left:3px solid #DC2626"' if is_violation else ""
            return f"""
<div class="col-inner"{border}>
  <div class="col-label">{cfg_label}</div>
  <div class="detail-block">
    <div class="detail-lbl">Sous-questions</div>
    {sq_list(sub)}
  </div>
  <div class="detail-block">
    <div class="detail-lbl">Sources mobilisées (type)</div>
    {sm_pills(sm)}
    <div class="reg-info muted small">{reg_info_display(ri)}</div>
  </div>
  <div class="detail-block">
    <div class="detail-lbl">Verdict registre</div>
    <div class="verdict-row">
      <span class="sig-lbl">(a) sources :</span> {verdict_badge(va)}
      <span class="sig-lbl" style="margin-left:12px">(b) LLM :</span> {verdict_badge(vb.get("verdict"))}
    </div>
    {f'<div class="b-detail muted small"><b>Extrait :</b> {b_extr[:100]}</div>' if b_extr else ''}
    {f'<div class="b-detail muted small"><b>Explication :</b> {b_expl[:120]}</div>' if b_expl else ''}
  </div>
  <div class="detail-block">
    <div class="detail-lbl">Réponse complète</div>
    <details class="ans-details"><summary class="ans-toggle">Afficher la réponse</summary>
    <div class="answer-box">{ans}</div></details>
  </div>
</div>"""

        is_vio_typ = typ_e.get("signal_a")=="VIOLATION" or typ_e.get("signal_b",{}).get("verdict")=="VIOLATION"
        is_vio_nt  = nt_e.get("signal_a")=="VIOLATION"  or nt_e.get("signal_b",{}).get("verdict")=="VIOLATION"
        header_cls = "section-bad" if (is_vio_typ or is_vio_nt) else ""

        detail_sections.append(f"""
<div class="q-section {header_cls}">
  <div class="q-header">
    <span class="q-num">Q{row}</span>
    <span class="q-orient-badge orient-{orientation}">{orient_lbl}</span>
    <span class="q-text">{question}</span>
  </div>
  <div class="q-cols">
    {col_html(typ_e, "v_decomp_raptor (TYPAGE)")}
    {col_html(nt_e,  "v_decomp_raptor_no_typing (NO_TYPING)")}
  </div>
</div>""")

    detail_html = "\n".join(detail_sections)

    # ── Tableau synthèse HTML ─────────────────────────────────────────────────
    def pct(n, d):
        return f"{100*n//d}%" if d else "—"

    summary_rows = []
    for r in results:
        row = r["row"]
        orient = r["orientation"]
        typ_e = r["configs"].get("v_decomp_raptor", {})
        nt_e  = r["configs"].get("v_decomp_raptor_no_typing", {})
        va_t  = typ_e.get("signal_a", "—")
        va_nt = nt_e.get("signal_a",  "—")
        vb_t  = (typ_e.get("signal_b") or {}).get("verdict", "—")
        vb_nt = (nt_e.get("signal_b")  or {}).get("verdict", "—")
        orient_lbl = {"objective":"Objectif","perception":"Perception","vaste":"Vaste"}[orient]
        summary_rows.append(
            f'<tr class="orient-{orient}">'
            f'<td class="num muted">Q{row}</td>'
            f'<td class="orient-cell">{orient_lbl}</td>'
            f'<td class="muted small">{r["question"][:55]}…</td>'
            f'<td class="center">{verdict_badge(va_t)}</td>'
            f'<td class="center">{verdict_badge(vb_t)}</td>'
            f'<td class="center">{verdict_badge(va_nt)}</td>'
            f'<td class="center">{verdict_badge(vb_nt)}</td>'
            f'</tr>'
        )
    summary_html = "\n".join(summary_rows)

    HTML = f"""<title>Test typage Raptor — {ts}</title>
<style>
:root {{
  --bg:#F1F3F7; --surface:#FFF; --surface-2:#F7F8FB; --border:#DDE1EC;
  --text:#1C2033; --muted:#6272A0; --accent:#7C3AED; --accent-dim:#EDE9FE;
  --pos:#16A34A; --neg:#DC2626; --warn:#B45309; --warn-bg:#FFFBEB;
  --shadow:0 1px 3px rgba(28,32,51,.08),0 4px 16px rgba(28,32,51,.04);
  --radius:6px;
  --font:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  --mono:ui-monospace,'SF Mono',Consolas,monospace;
}}
@media(prefers-color-scheme:dark){{
  :root{{--bg:#0D0F18;--surface:#161929;--surface-2:#1E2236;--border:#2A2F47;
  --text:#E8EAF6;--muted:#8892B0;--accent:#9B67F8;--accent-dim:#2A1F4F;
  --pos:#22C55E;--neg:#F87171;--warn:#FBBF24;--warn-bg:#1C1400;}}
}}
:root[data-theme=dark]{{--bg:#0D0F18;--surface:#161929;--surface-2:#1E2236;--border:#2A2F47;--text:#E8EAF6;--muted:#8892B0;--accent:#9B67F8;--accent-dim:#2A1F4F;--pos:#22C55E;--neg:#F87171;--warn:#FBBF24;--warn-bg:#1C1400;}}
:root[data-theme=light]{{--bg:#F1F3F7;--surface:#FFF;--surface-2:#F7F8FB;--border:#DDE1EC;--text:#1C2033;--muted:#6272A0;--accent:#7C3AED;--accent-dim:#EDE9FE;--pos:#16A34A;--neg:#DC2626;--warn:#B45309;--warn-bg:#FFFBEB;}}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{font-family:var(--font);font-size:13px;line-height:1.6;color:var(--text);background:var(--bg);min-height:100vh}}

.topnav{{position:sticky;top:0;z-index:100;background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 24px;display:flex;align-items:center;height:44px;box-shadow:0 1px 4px rgba(28,32,51,.06)}}
.nav-brand{{font-weight:700;font-size:12px;letter-spacing:.04em;color:var(--accent);text-transform:uppercase;margin-right:20px}}
.nav-links{{display:flex;flex:1;overflow-x:auto}}
.nav-links a{{padding:0 14px;height:44px;display:flex;align-items:center;font-size:12px;color:var(--muted);
  white-space:nowrap;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}}
.nav-links a:hover{{color:var(--text);text-decoration:none;border-color:var(--border)}}
.theme-btn{{margin-left:auto;padding:4px 10px;border-radius:4px;border:1px solid var(--border);
  background:var(--surface-2);color:var(--muted);cursor:pointer;font-size:11px}}

.page{{max-width:1200px;margin:0 auto;padding:32px 24px 80px}}
.section{{margin-bottom:48px;scroll-margin-top:56px}}
.section-eyebrow{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}}
.section-title{{font-size:18px;font-weight:700;color:var(--text);letter-spacing:-.02em;margin-bottom:16px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:20px 24px;margin-bottom:16px}}
.overflow-x{{overflow-x:auto}}

/* Verdict hero */
.verdict-hero{{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--accent);
  border-radius:var(--radius);padding:20px 24px;margin-bottom:28px;box-shadow:var(--shadow)}}
.verdict-hero .label{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:4px}}
.verdict-hero .headline{{font-size:17px;font-weight:700;letter-spacing:-.02em;line-height:1.3}}
.verdict-hero .sub{{margin-top:8px;color:var(--muted);font-size:12px}}

/* Summary table */
table{{border-collapse:collapse;width:100%;font-size:12px}}
th{{background:var(--surface-2);color:var(--muted);font-weight:600;font-size:10px;text-transform:uppercase;
  letter-spacing:.06em;padding:8px 10px;border-bottom:2px solid var(--border);white-space:nowrap;
  position:sticky;top:44px;z-index:10}}
td{{padding:6px 10px;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--surface-2)}}
.num{{font-variant-numeric:tabular-nums;font-family:var(--mono)}}
.center{{text-align:center}}
.muted{{color:var(--muted)}}
.small{{font-size:11px}}

/* orientation row colors */
tr.orient-objective{{}}
tr.orient-perception{{background:color-mix(in srgb, #7C3AED 4%, var(--surface))}}
tr.orient-vaste{{background:color-mix(in srgb, #2980B9 3%, var(--surface))}}
.orient-cell{{font-size:11px;color:var(--muted);white-space:nowrap}}

/* Badges */
.badge{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}}
.badge.good{{background:color-mix(in srgb,var(--pos) 14%,transparent);color:var(--pos)}}
.badge.bad{{background:color-mix(in srgb,var(--neg) 14%,transparent);color:var(--neg)}}
.badge.warn{{background:color-mix(in srgb,var(--warn) 14%,transparent);color:var(--warn)}}
.badge.na{{background:var(--surface-2);color:var(--muted)}}

/* Note */
.note{{background:var(--surface-2);border:1px solid var(--border);border-radius:4px;
  padding:10px 14px;color:var(--muted);font-size:11px;margin-bottom:16px;line-height:1.6}}
.note strong{{color:var(--text)}}
.note.warn{{background:var(--warn-bg);border-color:color-mix(in srgb,var(--warn) 40%,transparent);color:var(--warn)}}

/* Detail Q sections */
.q-section{{margin-bottom:32px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}}
.section-bad{{border-color:color-mix(in srgb,var(--neg) 30%,var(--border))}}
.q-header{{padding:12px 16px;background:var(--surface-2);border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
.q-num{{font-weight:700;color:var(--accent);font-size:14px;flex-shrink:0}}
.q-orient-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.05em;flex-shrink:0}}
.orient-objective .q-orient-badge,.q-orient-badge.orient-objective{{background:color-mix(in srgb,#C0392B 12%,transparent);color:#C0392B}}
.orient-perception .q-orient-badge,.q-orient-badge.orient-perception{{background:var(--accent-dim);color:var(--accent)}}
.orient-vaste .q-orient-badge,.q-orient-badge.orient-vaste{{background:color-mix(in srgb,#2980B9 12%,transparent);color:#2980B9}}
.q-text{{font-size:13px;color:var(--text)}}

.q-cols{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.col-inner{{padding:14px 16px;border-right:1px solid var(--border)}}
.col-inner:last-child{{border-right:none}}
.col-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:10px}}
.detail-block{{margin-bottom:12px}}
.detail-lbl{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px}}
.verdict-row{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px}}
.sig-lbl{{font-size:11px;color:var(--muted)}}
.b-detail{{margin-top:3px;line-height:1.4}}
.answer-box{{background:var(--surface-2);border:1px solid var(--border);border-radius:4px;
  padding:8px 10px;font-size:11px;line-height:1.6;color:var(--text);white-space:pre-wrap;word-break:break-word}}
.ans-details{{margin-top:4px}}
.ans-toggle{{font-size:11px;color:var(--accent);cursor:pointer;user-select:none;padding:2px 0}}
.ans-toggle::-webkit-details-marker{{color:var(--accent)}}
.col-error{{padding:14px;color:var(--neg);font-size:12px}}

.sq-list{{margin:0;padding-left:18px;font-size:12px;line-height:1.5}}
.sq-list li{{margin-bottom:6px}}
.sq-idx{{font-weight:600;color:var(--accent);margin-right:4px}}
.sq-q{{font-weight:500}}
.sq-ans-details{{margin-top:3px;margin-left:16px}}
.sq-ans-toggle{{font-size:11px;color:var(--muted);cursor:pointer;user-select:none}}
.sq-ans-toggle::-webkit-details-marker{{color:var(--muted)}}
.sq-ans-body{{background:var(--surface-2);border-left:2px solid var(--border);
  margin-top:4px;padding:6px 8px;font-size:11px;line-height:1.6;color:var(--text);
  white-space:pre-wrap;word-break:break-word}}

.pill-quali{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;margin:2px 2px;
  background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)}}
.pill-quanti{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;margin:2px 2px;
  background:color-mix(in srgb,#C0392B 14%,transparent);color:#C0392B}}
.pill-mixed{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;margin:2px 2px;
  background:color-mix(in srgb,#E07020 14%,transparent);color:#E07020}}
.reg-info{{margin-top:4px}}

.stat-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px 18px;flex:1;min-width:120px;box-shadow:var(--shadow)}}
.sc-label{{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}}
.sc-val{{font-size:22px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums;margin:2px 0 1px}}
.sc-sub{{font-size:11px;color:var(--muted)}}
.section + .section{{border-top:1px solid var(--border);padding-top:40px;margin-top:0}}
</style>

<nav class="topnav">
  <span class="nav-brand">Test Typage Raptor</span>
  <div class="nav-links">
    <a href="#synthese">Synthèse</a>
    <a href="#detail">Détail Q×Q</a>
  </div>
  <button class="theme-btn" onclick="toggleTheme()">◐ Thème</button>
</nav>

<div class="page">

<div id="synthese" class="section">
<div class="section-eyebrow">Résultat principal</div>
<div class="section-title">Effet de la règle de typage sur Decomp+Raptor</div>

<div class="verdict-hero">
  <div class="label">Verdict</div>
  <div class="headline" style="color:{verdict_col}">{verdict_txt}</div>
  <div class="sub">
    12 questions · 2 configs · Signal (a) déterministe (sources) + Signal (b) LLM étroit (GPT-4o-mini) · {ts}<br>
    <b>Fast-path global Corse</b> : aucune des 12 questions ne le déclenche (vérifié) — les deux variantes passent par decompose_question().
  </div>
</div>

<div class="stat-row">
  <div class="stat-card">
    <div class="sc-label">Violations (a) — Typage</div>
    <div class="sc-val" style="color:{'var(--neg)' if sa_typ_or else 'var(--pos)'}">{sa_typ_or}/{n_oriented}</div>
    <div class="sc-sub">questions orientées</div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Violations (a) — No_Typing</div>
    <div class="sc-val" style="color:{'var(--neg)' if sa_nt_or else 'var(--pos)'}">{sa_nt_or}/{n_oriented}</div>
    <div class="sc-sub">questions orientées</div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Violations (b) — Typage</div>
    <div class="sc-val" style="color:{'var(--neg)' if sb_typ_or else 'var(--pos)'}">{sb_typ_or}/{n_oriented}</div>
    <div class="sc-sub">questions orientées</div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Violations (b) — No_Typing</div>
    <div class="sc-val" style="color:{'var(--neg)' if sb_nt_or else 'var(--pos)'}">{sb_nt_or}/{n_oriented}</div>
    <div class="sc-sub">questions orientées</div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Incomplets vastes — Typage</div>
    <div class="sc-val">{sa_typ_va}/{n_vastes}</div>
    <div class="sc-sub">contrôle (Δ = {sa_nt_va - sa_typ_va:+d})</div>
  </div>
</div>

<div class="note">
  <strong>Signal (a) :</strong> classification déterministe depuis <code>sources_mobilisees</code> (auto-déclaration Mistral via le bloc <code>===SOURCES_MOBILISEES===</code>).
  VIOLATION = question objective qui mobilise substantiellement du QUALI (ou perception → QUANTI).
  INCOMPLET = question vaste sans mix des deux familles.<br>
  <strong>Signal (b) :</strong> GPT-4o-mini ciblé (1 appel par réponse orientée, ~100 tokens) — détecte si la réponse contient substantiellement du registre opposé.
  N/A pour les questions vastes (les deux registres sont légitimes).
</div>

<div class="overflow-x">
<table>
<tr>
  <th>#</th><th>Orient.</th><th>Question</th>
  <th colspan="2" style="background:color-mix(in srgb,#27AE60 8%,var(--surface-2))">TYPAGE</th>
  <th colspan="2" style="background:color-mix(in srgb,#7C3AED 8%,var(--surface-2))">NO_TYPING</th>
</tr>
<tr>
  <th></th><th></th><th></th>
  <th style="background:color-mix(in srgb,#27AE60 8%,var(--surface-2))">Sig.(a)</th>
  <th style="background:color-mix(in srgb,#27AE60 8%,var(--surface-2))">Sig.(b)</th>
  <th style="background:color-mix(in srgb,#7C3AED 8%,var(--surface-2))">Sig.(a)</th>
  <th style="background:color-mix(in srgb,#7C3AED 8%,var(--surface-2))">Sig.(b)</th>
</tr>
{summary_html}
</table>
</div>
</div>

<div id="detail" class="section">
<div class="section-eyebrow">Section 2</div>
<div class="section-title">Détail côte à côte par question</div>

<div class="note">
  <span style="color:#C0392B">■</span> Objectif (QUANTI attendu) &nbsp;
  <span style="color:var(--accent)">■</span> Perception (QUALI attendu) &nbsp;
  <span style="color:#2980B9">■</span> Vaste (deux familles, témoins)
  &nbsp;·&nbsp; Bordure rouge gauche = au moins une violation détectée.
</div>

{detail_html}
</div>

<p style="color:var(--muted);font-size:11px;margin-top:32px">
  Généré le {ts} · Commande : <code>python test_typing_raptor.py</code>
  · Configs : v_decomp_raptor (use_bilan=False) vs v_decomp_raptor_no_typing (use_bilan=False, no_typing=True)
  · Signal (b) : GPT-4o-mini
</p>
</div>

<script>
function toggleTheme() {{
  const root = document.documentElement;
  const cur  = root.getAttribute('data-theme');
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (!cur) root.setAttribute('data-theme', dark ? 'light' : 'dark');
  else if (cur === 'dark') root.setAttribute('data-theme', 'light');
  else root.removeAttribute('data-theme');
}}
</script>
"""
    return HTML

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Regen HTML depuis JSONs existants")
    args = parser.parse_args()

    if args.report:
        results = load_results()
        if not results:
            sys.exit("Aucun résultat trouvé dans " + str(OUT_DIR))
        print("Résultats chargés depuis JSONs existants.")
    else:
        results = run_all()

    html = make_html(results)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size = HTML_PATH.stat().st_size
    print(f"\nHTML → {HTML_PATH}  ({size//1024} Ko)")
    print(f"Pour rejouer : python test_typing_raptor.py")
    print(f"Pour regen HTML seulement : python test_typing_raptor.py --report")
