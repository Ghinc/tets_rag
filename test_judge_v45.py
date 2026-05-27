"""
Test du juge V4.5 sur Q45, Q49, Q50 — toutes configs.
V4.5 = V4.4 + Exemple 9 (comparaison avec données partiellement absentes signalées)
Produit un HTML de récap comparatif V4.3 / V4.4 / V4.5.
"""
import json, re, sys, io, glob, html as html_mod
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import _JUDGE_V43_SYSTEM, _call_llm, _build_sources_text

# ── Construire V4.4 puis V4.5 ─────────────────────────────────────────────────
_RULE3_V43 = """\
### Règle 3 — Absence signalée de source qualitative

Si la question demande un croisement quali/quanti (perceptions vs
indicateurs, ressenti vs structurel) ET qu'aucune source qualitative
(verbatim, entretien, raptor_quali) n'apparaît dans les sources
fournies au système :

- Si la réponse signale explicitement le manque ("aucune donnée
  qualitative disponible pour ce sous-groupe") → comportement
  acceptable, pas de pénalité
- Si la réponse ne signale pas le manque et présente quand même un
  croisement (en renommant le quanti en quali, par exemple) → ÉCHEC
  MAJEUR :
  - `pertinence` ≤ 3
  - `fondement_factuel` ≤ 3
  - `nuance_incertitude` ≤ 2"""

_RULE3_V44 = """\
### Règle 3 — Mislabelling silencieux uniquement

Cette règle pénalise UNIQUEMENT le mislabelling silencieux : présenter
des données quantitatives comme qualitatives sans aucune mention de
la limite.

Conditions de déclenchement strictes :
- (1) La question demande un croisement quali/quanti OU des perceptions
- (2) Aucune source qualitative dans les sources fournies au système
- (3) La réponse présente des données quantitatives en les appelant
      qualitatives (ex : "voici les dimensions qualitatives OppChoVec")
- (4) AUCUN signalement de l'absence de données qualitatives nulle
      part dans la réponse (ni en titre, ni dans le texte, ni dans
      un tableau, ni en synthèse)

Les 4 conditions doivent être réunies pour déclencher la pénalité.

Si la réponse signale le manque de données qualitatives DE QUELQUE
MANIÈRE QUE CE SOIT (titre de section "Données manquantes", phrase
"aucune donnée disponible", tableau marquant "Inconnu", mention "ne
permettent pas de"), même brève, la Règle 3 NE S'APPLIQUE PAS. La
réponse mobilise alors des données contextuelles avec transparence,
ce qui est conforme au Principe 2 (transparence rachète l'incomplétude).

Pénalités si Règle 3 déclenchée (4 conditions réunies) :
- `pertinence` ≤ 3
- `fondement_factuel` ≤ 3
- `nuance_incertitude` ≤ 2"""

_STEP3_V43 = """\
**Étape 3** : confronte les affirmations aux sources. VÉRIFIE les
4 règles anti-mislabelling :
- La réponse renomme-t-elle une source quanti en quali ?
- La réponse surinterprète-t-elle Vec/Cho/Opp ?
- La question demande-t-elle un croisement quali/quanti sans source
  quali fournie ?
- La réponse extrapole-t-elle OppChoVec à un sous-groupe ?"""

_STEP3_V44 = """\
**Étape 3** : confronte les affirmations aux sources. VÉRIFIE les
4 règles anti-mislabelling :
- La réponse renomme-t-elle une source quanti en quali ?
- La réponse surinterprète-t-elle Vec/Cho/Opp ?
- La question demande-t-elle un croisement quali/quanti sans source
  quali fournie ?
- La réponse extrapole-t-elle OppChoVec à un sous-groupe ?

**Étape 3.bis — Recherche EXHAUSTIVE des signalements d'absence**

Si la question demande des perceptions ou un croisement quali/quanti
et que les sources contiennent uniquement du quantitatif, AVANT de
déclencher la Règle 3, scanne la réponse entière pour identifier
TOUT signalement d'absence de données qualitatives, sous N'IMPORTE
QUELLE forme :

- Titres de section ("Données manquantes", "Inconnu pour X")
- Phrases dans le texte ("aucune donnée d'enquête disponible", "ne
  permettent pas de répondre directement", "se limite aux indicateurs
  objectifs")
- Cellules de tableau ("Inconnu", "Données indisponibles", "—")
- Recommandations finales ("une enquête locale serait nécessaire")
- Modalisateurs explicites ("ce qui suggère", "pourrait", "il
  conviendrait de confirmer")

Un SEUL signalement clair suffit à ne pas déclencher la Règle 3.

Documente dans le champ `signalements_detectes` les signalements
trouvés (ou leur absence)."""

_MISLABELLING_JSON_V43 = """\
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non | oui — <détail>",
    "regle_2_surinterpretation_oppchovec": "non | oui — <détail>",
    "regle_3_absence_quali_non_signalee": "non | oui — <détail>",
    "regle_4_extrapolation_sous_groupe": "non | oui — <détail>"
  },"""

_MISLABELLING_JSON_V44 = """\
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non | oui — <détail>",
    "regle_2_surinterpretation_oppchovec": "non | oui — <détail>",
    "regle_3_absence_quali_non_signalee": "non | oui — <détail>",
    "regle_4_extrapolation_sous_groupe": "non | oui — <détail>"
  },
  "signalements_detectes": [
    "<liste des signalements d'absence de données quali trouvés dans la réponse, ou [] si aucun>"
  ],"""

_EXAMPLE9 = """
--- EXEMPLE 9 : Comparaison avec données partiellement absentes signalées ---

QUESTION (fictive) : Comment se compare la satisfaction concernant les
transports entre Bastia et Piedicorte-di-Gaggio ?

SECTION : Raisonnement comparatif / Croisé quanti/quali

SOURCES FOURNIES (fictives) : oppchovec:Bastia (score mobilité 7.1),
raptor:commune_quali:Bastia (verbatims sur transports), oppchovec:Piedicorte
(score mobilité 8.9). Aucun verbatim ni entretien pour Piedicorte.

RÉPONSE (extrait) : "À Bastia, les habitants expriment une satisfaction
modérée concernant les transports : si la desserte urbaine est jugée
correcte, plusieurs verbatims pointent une saturation aux heures de pointe
et des difficultés de stationnement. Les indicateurs territoriaux
confirment ce constat nuancé (score mobilité 7.1/10).

Concernant Piedicorte-di-Gaggio, les indicateurs objectifs sont
disponibles (score mobilité 8.9/10, lié à la position géographique et
à l'accès routier). En revanche, aucune donnée d'enquête citoyenne
n'est disponible pour cette commune, ce qui empêche d'évaluer la
satisfaction perçue par les habitants. La comparaison entre les deux
communes ne peut donc être menée que sur le plan des indicateurs
territoriaux ; les perceptions resteront à confirmer par une enquête
locale à Piedicorte."

NOTATION :
{
  "section_observee": "Raisonnement comparatif / Croisé quanti/quali",
  "comportement_attendu_selon_grille": "Comparaison effectuée sur les éléments disponibles avec signalement explicite des manques",
  "type_reponse_attendue_specifie": "reponse_substantielle_attendue",
  "coherence_comportement_observe": "oui",
  "sources_inventaire": [
    {"source": "oppchovec:Bastia", "type": "quanti"},
    {"source": "raptor:commune_quali:Bastia", "type": "quali"},
    {"source": "oppchovec:Piedicorte", "type": "quanti"}
  ],
  "signalements_detectes": [
    "aucune donnée d'enquête citoyenne n'est disponible pour cette commune",
    "ce qui empêche d'évaluer la satisfaction perçue par les habitants",
    "la comparaison ne peut donc être menée que sur le plan des indicateurs territoriaux",
    "les perceptions resteront à confirmer par une enquête locale"
  ],
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non",
    "regle_2_surinterpretation_oppchovec": "non",
    "regle_3_absence_quali_non_signalee": "non — signalements multiples et explicites de l'absence de données quali pour Piedicorte",
    "regle_4_extrapolation_sous_groupe": "non"
  },
  "elements_specifiques_question": ["transports", "Bastia", "Piedicorte-di-Gaggio", "satisfaction", "comparaison"],
  "elements_traitement": [
    {"element": "transports Bastia", "traitement": "precis"},
    {"element": "transports Piedicorte", "traitement": "approximation_signalee"},
    {"element": "satisfaction perçue Piedicorte", "traitement": "omis — absence signalée"}
  ],
  "raisonnement": "Comparaison partielle menée honnêtement : Bastia quanti+quali, Piedicorte quanti seul avec signalement explicite du manque quali. Règle 3 non déclenchée.",
  "pertinence": {"note": 4, "justification": "Comparaison correctement menée sur les données disponibles, avec délimitation honnête du périmètre comparable."},
  "fondement_factuel": {"note": 5, "justification": "Affirmations bien sourcées sur Bastia, périmètre des données Piedicorte explicité, aucune invention."},
  "nuance_incertitude": {"note": 5, "justification": "Excellent signalement des limites : absence quali pour Piedicorte mentionnée plusieurs fois, recommandation d'enquête locale."},
  "coherence_qualiquanti": {"note": 4, "justification": "Mobilisation appropriée des deux familles pour Bastia, restriction transparente au seul quanti pour Piedicorte."}
}

PRINCIPE ILLUSTRÉ : quand une comparaison porte sur deux entités dont
l'une manque de données qualitatives, le bon comportement est de
mener la comparaison sur les éléments disponibles ET de signaler
explicitement, à plusieurs reprises dans la réponse, ce qui ne peut
pas être comparé. La Règle 3 (mislabelling silencieux) NE doit PAS
être déclenchée tant que les signalements sont présents, même
dispersés dans le texte ou dans des sections différentes.
"""

_FINAL_MARKER = "=== MAINTENANT, ÉVALUE LA RÉPONSE SUIVANTE ==="

_JUDGE_V44_SYSTEM = (
    _JUDGE_V43_SYSTEM
    .replace(_RULE3_V43, _RULE3_V44)
    .replace(_STEP3_V43, _STEP3_V44)
    .replace(_MISLABELLING_JSON_V43, _MISLABELLING_JSON_V44)
)
_JUDGE_V45_SYSTEM = _JUDGE_V44_SYSTEM.replace(
    _FINAL_MARKER,
    _EXAMPLE9 + "\n" + _FINAL_MARKER
)

assert "EXEMPLE 9" in _JUDGE_V45_SYSTEM
print(f"Prompts OK — V4.3:{len(_JUDGE_V43_SYSTEM)} V4.4:{len(_JUDGE_V44_SYSTEM)} V4.5:{len(_JUDGE_V45_SYSTEM)}")

# ── Parser (commun V4.4/V4.5) ─────────────────────────────────────────────────
def _parse(j):
    def note(d):
        return d.get("note") if isinstance(d, dict) else (d if isinstance(d, (int, float)) else None)
    p, ff, ni, qq = note(j.get("pertinence")), note(j.get("fondement_factuel")), note(j.get("nuance_incertitude")), note(j.get("coherence_qualiquanti"))
    scores = [x for x in [p, ff, ni, qq] if x is not None]
    sg = round(sum(scores)/len(scores), 4) if scores else None
    mis = j.get("mislabelling_detecte", {})
    return {
        "pertinence": p, "fondement_factuel": ff, "nuance_incertitude": ni, "coherence_qualiquanti": qq,
        "score_global": sg,
        "mislabelling_flag": any(str(v).lower() not in ("non","false","","null","none") for v in mis.values()),
        "mislabelling_detecte": mis,
        "signalements_detectes": j.get("signalements_detectes", []),
        "raisonnement": j.get("raisonnement", ""),
        "pertinence_justif":          j.get("pertinence",{}).get("justification","") if isinstance(j.get("pertinence"),dict) else "",
        "fondement_factuel_justif":   j.get("fondement_factuel",{}).get("justification","") if isinstance(j.get("fondement_factuel"),dict) else "",
        "nuance_incertitude_justif":  j.get("nuance_incertitude",{}).get("justification","") if isinstance(j.get("nuance_incertitude"),dict) else "",
        "coherence_qualiquanti_justif":j.get("coherence_qualiquanti",{}).get("justification","") if isinstance(j.get("coherence_qualiquanti"),dict) else "",
    }

def run_judge(prompt_sys, question, answer, sources, section, subsection, expected_type):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\nSECTION : {section}\n\nSOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\nSOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\nÉvalue cette réponse selon la procédure et le format spécifiés."
    )
    try:
        raw = _call_llm(prompt_sys, user_prompt, max_tokens=3500, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        return _parse(j)
    except Exception as e:
        return {"judge_error": str(e), "score_global": None}

# ── Charger données ───────────────────────────────────────────────────────────
with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    complet = json.load(f)

f44_files = sorted(glob.glob('comparaisons_rag/test_v44_*.json'))
v44_data = json.load(open(f44_files[-1], encoding='utf-8')) if f44_files else {"results": {}}

by_row = {}
for ver, entries in complet.items():
    for e in entries:
        row = e['excel_row']
        if row not in by_row: by_row[row] = {}
        by_row[row][ver] = e

CONFIGS      = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
CFG_LABELS   = {'v_vanilla_k10':'Vanilla k10','v_vanilla_k25':'Vanilla k25','v_decomp':'Decomp','v_decomp_raptor':'Decomp+RAPTOR'}
TARGET_ROWS  = [45, 49, 50]

# ── Lancer V4.5 ───────────────────────────────────────────────────────────────
results_v45 = {}
print(f"\nLancement V4.5 — {len(TARGET_ROWS)*len(CONFIGS)} appels juge\n{'='*65}")

for row in TARGET_ROWS:
    results_v45[row] = {}
    vmap = by_row.get(row, {})
    ref  = next(iter(vmap.values()), {})
    q    = ref.get('question','')
    sec  = ref.get('section','')
    sub  = ref.get('subsection','')
    exp  = ref.get('type_reponse_attendue_specifie','reponse_substantielle_attendue') or 'reponse_substantielle_attendue'

    print(f"\nQ{row} — {q[:70]}")
    for cfg in CONFIGS:
        e = vmap.get(cfg, {})
        ans = e.get('answer','') or ''
        src = e.get('sources',[]) or []
        if not ans:
            results_v45[row][cfg] = None
            continue
        r = run_judge(_JUDGE_V45_SYSTEM, q, ans, src, sec, sub, exp)
        results_v45[row][cfg] = r
        sg43 = e.get('score_global')
        sg44 = ((v44_data['results'].get(str(row)) or {}).get(cfg) or {}).get('score_global')
        sg45 = r.get('score_global')
        mis  = '✗' if r.get('mislabelling_flag') else ' '
        d43  = f"{sg45-sg43:+.2f}" if sg45 is not None and sg43 is not None else " n/a"
        sigs = r.get('signalements_detectes',[])
        r3   = (r.get('mislabelling_detecte') or {}).get('regle_3_absence_quali_non_signalee','?')[:60]
        print(f"  [{cfg:20s}] V4.3={sg43} V4.4={sg44} V4.5={sg45}{mis}  Δ43={d43}")
        print(f"  {' ':22s} R3={r3}")
        print(f"  {' ':22s} Sig={sigs[:2]}")

# ── Sauvegarder JSON ──────────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_json = f'comparaisons_rag/test_v45_{ts}.json'
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({'metadata': {'rows': TARGET_ROWS, 'configs': CONFIGS, 'judge': 'v4.5', 'ts': ts},
               'results': {str(k): v for k, v in results_v45.items()}}, f, ensure_ascii=False, indent=2)
print(f"\nJSON -> {out_json}")

# ── HTML ──────────────────────────────────────────────────────────────────────
def sc(v):
    if v is None: return '#9e9e9e'
    return '#388e3c' if v >= 4.5 else ('#f57c00' if v >= 3.5 else '#c62828')

def badge(v, mis=False):
    if v is None: return '<span style="color:#aaa">—</span>'
    mis_s = '<span style="background:#c62828;color:white;font-size:0.65em;padding:1px 4px;border-radius:3px;margin-left:3px">MIS</span>' if mis else ''
    return f'<span style="background:{sc(v)};color:white;padding:2px 8px;border-radius:10px;font-weight:bold;font-size:0.88em">{v:.2f}</span>{mis_s}'

def delta_cell(a, b):
    if a is None or b is None: return '<td style="text-align:center;color:#aaa">—</td>'
    d = b - a
    col = '#388e3c' if d > 0.1 else ('#c62828' if d < -0.1 else '#666')
    return f'<td style="text-align:center;color:{col};font-weight:bold">{d:+.2f}</td>'

CSS = """
body{font-family:Arial,sans-serif;font-size:14px;max-width:1350px;margin:0 auto;padding:20px;background:#f5f5f5}
h1{border-bottom:3px solid #444;padding-bottom:8px;color:#222}
h2{background:#ddd;padding:6px 14px;border-radius:4px;font-size:1em;color:#333;margin:28px 0 8px}
.qblock{background:white;border-radius:8px;padding:18px;margin-bottom:24px;box-shadow:0 1px 5px rgba(0,0,0,.1)}
.qtitle{font-size:1.05em;font-weight:bold;color:#111;margin-bottom:4px}
.sub{display:inline-block;background:#555;color:white;font-size:.72em;padding:2px 8px;border-radius:10px;margin-bottom:12px}
table{border-collapse:collapse;width:100%;margin-bottom:12px}
th{background:#444;color:white;padding:6px 10px;font-size:.82em;text-align:center}
td{border:1px solid #ddd;padding:5px 9px;font-size:.82em}
.cfg{border-left:4px solid #bbb;padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0}
.cfg-label{font-weight:bold;font-size:.88em;margin-bottom:3px}
.mini{font-size:.75em;color:#777;margin-bottom:3px}
.judge-r{font-size:.8em;color:#444;font-style:italic;border-left:2px solid #ddd;padding-left:8px;margin:4px 0}
.sig{font-size:.78em;color:#1565c0;margin-top:3px}
details summary{cursor:pointer;font-size:.83em;color:#1565c0;margin-top:4px}
.answer-box{font-size:.82em;color:#333;line-height:1.5;max-height:300px;overflow-y:auto;border:1px solid #ddd;padding:8px;background:#fafafa;border-radius:4px;white-space:pre-wrap;margin-top:5px}
"""

lines = ['<!DOCTYPE html>','<html lang="fr">','<head>','<meta charset="UTF-8">',
         f'<title>Test V4.5 — Q45 Q49 Q50</title>',
         f'<style>{CSS}</style>','</head>','<body>',
         '<h1>Test juge V4.5 — Q45, Q49, Q50 (toutes configs)</h1>',
         f'<p style="color:#666">Juge GPT-4o · V4.5 = V4.4 + Exemple 9 · {ts}</p>']

for row in TARGET_ROWS:
    vmap  = by_row.get(row, {})
    ref   = next(iter(vmap.values()),{})
    q     = ref.get('question','')
    sub   = ref.get('subsection','')
    sec   = ref.get('section','')

    lines.append(f'<h2>{html_mod.escape(sec)}</h2>')
    lines.append(f'<div class="qblock" id="q{row}">')
    lines.append(f'<div class="qtitle">Q{row} — {html_mod.escape(q)}</div>')
    lines.append(f'<span class="sub">{html_mod.escape(sub)}</span>')

    # Scores comparison table
    lines.append('<table><tr><th>Config</th><th>V4.3</th><th>V4.4</th><th>V4.5</th><th>Δ V4.3→V4.5</th><th>Δ V4.4→V4.5</th><th>R3 V4.5</th></tr>')
    for cfg in CONFIGS:
        e43  = vmap.get(cfg,{})
        e44  = ((v44_data['results'].get(str(row)) or {}).get(cfg)) or {}
        e45  = results_v45[row].get(cfg) or {}
        sg43 = e43.get('score_global')
        sg44 = e44.get('score_global')
        sg45 = e45.get('score_global')
        mis43= e43.get('mislabelling_flag',False)
        mis44= e44.get('mislabelling_flag',False)
        mis45= e45.get('mislabelling_flag',False)
        r3   = (e45.get('mislabelling_detecte') or {}).get('regle_3_absence_quali_non_signalee','—')
        r3_short = ('NON' if r3.startswith('non') else 'OUI') if r3 != '—' else '—'
        r3_col   = '#388e3c' if r3_short == 'NON' else '#c62828'
        lines.append(
            f'<tr><td><b>{CFG_LABELS[cfg]}</b></td>'
            f'<td style="text-align:center">{badge(sg43,mis43)}</td>'
            f'<td style="text-align:center">{badge(sg44,mis44)}</td>'
            f'<td style="text-align:center">{badge(sg45,mis45)}</td>'
            + delta_cell(sg43,sg45) + delta_cell(sg44,sg45) +
            f'<td style="text-align:center;color:{r3_col};font-weight:bold">{r3_short}</td></tr>'
        )
    lines.append('</table>')

    # Per-config detail blocks
    cfg_colors = {'v_vanilla_k10':'#e8f4f8','v_vanilla_k25':'#e8f0e8','v_decomp':'#f8f0e8','v_decomp_raptor':'#f0e8f8'}
    for cfg in CONFIGS:
        e45  = results_v45[row].get(cfg) or {}
        mis  = e45.get('mislabelling_flag',False)
        sg   = e45.get('score_global')
        rais = e45.get('raisonnement','')
        sigs = e45.get('signalements_detectes',[])
        ans  = (by_row[row].get(cfg,{}).get('answer','') or '').strip()
        bord = '#c62828' if mis else '#bdbdbd'

        lines.append(f'<div class="cfg" style="background:{cfg_colors[cfg]};border-left-color:{bord}">')
        mis_s = '<span style="background:#c62828;color:white;font-size:.68em;padding:1px 5px;border-radius:3px;margin-left:5px">MISLABELLING</span>' if mis else ''
        lines.append(f'<div class="cfg-label">{CFG_LABELS[cfg]} &nbsp; {badge(sg,mis)} {mis_s}</div>')
        p,ff,ni,qq = e45.get('pertinence'), e45.get('fondement_factuel'), e45.get('nuance_incertitude'), e45.get('coherence_qualiquanti')
        def fv(x): return f'{x:.2f}' if x is not None else '—'
        lines.append(f'<div class="mini">P={fv(p)} · F={fv(ff)} · N={fv(ni)} · Q/Q={fv(qq)}</div>')
        if sigs:
            sigs_html = ' · '.join(f'"{html_mod.escape(s[:80])}"' for s in sigs[:4])
            lines.append(f'<div class="sig">&#128269; Signalements : {sigs_html}</div>')
        else:
            lines.append('<div class="sig" style="color:#c62828">&#9888; Aucun signalement détecté</div>')
        if rais:
            lines.append(f'<div class="judge-r">&#129516; {html_mod.escape(rais)}</div>')
        if ans:
            lines.append('<details><summary>Voir la réponse RAG complète</summary>')
            lines.append(f'<div class="answer-box">{html_mod.escape(ans)}</div>')
            lines.append('</details>')
        lines.append('</div>')

    lines.append('</div>')  # qblock

lines += ['</body>','</html>']
out_html = f'comparaisons_rag/test_v45_{ts}.html'
with open(out_html, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"HTML -> {out_html}")
