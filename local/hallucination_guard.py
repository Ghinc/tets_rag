"""
hallucination_guard.py — 3 contrôles déterministes post-génération (zéro token).

Rule 1 — Entity silence
    Si une entité-cible n'a aucune donnée (toutes SQ = N/D en phase1,
    ou aucune source qualitative en baseline), elle passe en liste_silence.
    Le contexte de synthèse porte la liste explicitement (injection pré-synthèse).
    Post-check : toute phrase mentionnant l'entité sans marqueur d'absence → violation.

Rule 2 — Number grounding
    Chaque valeur numérique significative de la réponse finale doit apparaître
    (forme normalisée) dans le contexte injecté au synthétiseur.
    Fail loudly : log, ne relance pas.

Rule 3 — Quote grounding
    Toute chaîne entre guillemets > 5 mots doit se retrouver littéralement
    dans les extraits des sources récupérées.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ── Constantes ───────────────────────────────────────────────────────────────

_ABSENCE_RE = re.compile(
    r"pas de donn|aucune donn|absence de|non disponible|indisponible|"
    r"donn[ée]es inexistant|ne dispose pas|ne permet pas|sans donn[ée]e|"
    r"donn[ée]es absentes|manque de donn|lacune|inconnue|"
    r"il n['\']exist|il n['\']y a pas de donn|il est impossible|"
    r"donn[ée]es (?:manquant|insuffisant|limit)|"
    r"\bN/D\b|\bN\.D\b|données limitées|"
    # Constats d'absence d'entité (méta-énoncés du LLM pour expliciter le vide)
    r"sans mention de|sans information sur|aucune mention|aucune information sur|"
    r"n['\']est pas mentionn|n['\']apparai|non mentionn|"
    r"pas couverte?|non couverte?|pas repr[eé]sent|non repr[eé]sent|"
    r"ne concerne(?:nt)? pas|font d[eé]faut|ne permettent? pas|ne sont pas disponible|"
    r"ne couvre(?:nt)? pas|n['\']ont pas [eé]t[eé]|pas [eé]t[eé] recueilli|"
    r"ne figure(?:nt)? pas|ne port(?:e|ent) pas|"
    # Formes singulières et tournures observées sur les cas réels
    r"n['\']est pas disponible|ne contien(?:t|nent) pas|"
    r"restent? limit|sont limit|"
    r"proviennent? d['\']autres|d['\']autres communes|"
    r"sans d[eé]sagr[eé]gation|sp[eé]cifique(?:s)? [àa] la commune|"
    r"aucun(?:e)? (?:donn[eé]e|verbatim|information|source|enqu[eê]te) (?:sp[eé]cifique|disponible|recueilli)|"
    r"ne couvrent? pas|non attribu[eé]",
    re.IGNORECASE,
)

# Nombres significatifs : flottants, pourcentages, fractions X/N, entiers >= 10
# On EXCLUT : entiers < 10 sans unité, années (19xx-20xx)
_NUM_RE = re.compile(
    r'\b(\d+(?:[,\.]\d+)?)'     # nombre de base (int ou float)
    r'(?:\s*%|\s*/\s*\d+)?'     # optionnel : % ou /N
    r'\b',
    re.UNICODE,
)

# Guillemets français et anglais (capturer le contenu)
_QUOTE_RE = re.compile(r'[«""](.*?)[»""]', re.DOTALL)

# Langage verbatim/qualitatif — pour le check quali-only silence
_QUALI_LANG_RE = re.compile(
    r"habitant|citoyen|résident|verbatim|ressenti|perçoi|exprim|selon les enquêt|"
    r"témoign|avis des|dans leur vie|selon les habitants|les gens|la population ressentent|"
    r"les gens décrivent|rapporte|ont indiqué|ont cité|ont exprimé|ont mentionné|"
    r"enquête.*qualit|qualit.*enquête",
    re.IGNORECASE,
)


# ── Types de résultats ───────────────────────────────────────────────────────

@dataclass
class Violation:
    rule: str       # "silence" | "number" | "quote"
    detail: str
    evidence: str   # ce qui a été trouvé
    context: str    # ce qui était attendu / ce qui manque


@dataclass
class GuardReport:
    silence_list: list[str] = field(default_factory=list)
    # {commune: {"quali": bool, "quanti": bool}} — issu de build_silence_map_from_chroma
    silence_map: dict = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "silence_list": self.silence_list,
            "silence_map": self.silence_map,
            "n_violations": len(self.violations),
            "violations": [
                {
                    "rule": v.rule,
                    "detail": v.detail,
                    "evidence": v.evidence[:200],
                    "context": v.context[:200],
                }
                for v in self.violations
            ],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _nfkd(s: str) -> str:
    """Normalise accents → ASCII pour comparaisons tolérantes."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _normalize_num(s: str) -> str:
    """6,8 → 6.8 ; supprime espaces."""
    return s.replace(",", ".").replace(" ", "").strip()


def _is_year(s: str) -> bool:
    try:
        n = int(s.replace(",", "").replace(".", ""))
        return 1900 <= n <= 2100
    except ValueError:
        return False


def _is_trivial_int(s: str) -> bool:
    """Entier < 10 sans décimale → skip (trop commun, pas un point de données)."""
    if "." in s or "," in s:
        return False
    try:
        return int(s) < 10
    except ValueError:
        return False


# ── Parsing de blocs structurés ─────────────────────────────────────────────

_VERDICT_RE = re.compile(
    r'\[\[VERDICT\]\](.*?)\[\[/VERDICT\]\]', re.DOTALL | re.IGNORECASE
)
_CONTEXTE_RE = re.compile(
    r'\[\[CONTEXTE_ELARGI[^\]]*\]\](.*?)\[\[/CONTEXTE_ELARGI\]\]',
    re.DOTALL | re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(
    r'\[\[/?(?:VERDICT|CONTEXTE_ELARGI)[^\]]*\]\]', re.IGNORECASE
)


def _parse_blocks(text: str) -> dict[str, str]:
    """
    Découpe text en zones : verdict, contexte_elargi, outer (hors blocs).
    Retourne {'verdict': str, 'contexte_elargi': str, 'outer': str}.
    """
    verdict_m   = _VERDICT_RE.search(text)
    contexte_m  = _CONTEXTE_RE.search(text)

    verdict_text   = verdict_m.group(1).strip()  if verdict_m  else ""
    contexte_text  = contexte_m.group(1).strip() if contexte_m else ""

    # outer = texte hors balises (on retire les blocs entiers)
    outer = _ANY_TAG_RE.sub("", text)
    if verdict_m:
        outer = outer.replace(verdict_text, "", 1)
    if contexte_m:
        outer = outer.replace(contexte_text, "", 1)
    outer = outer.strip()

    return {"verdict": verdict_text, "contexte_elargi": contexte_text, "outer": outer}


def strip_block_tags(text: str) -> str:
    """Retire les balises de blocs avant affichage."""
    return _ANY_TAG_RE.sub("", text).strip()


# ── Règle 1 : entity silence ─────────────────────────────────────────────────

def build_silence_list_phase1(
    sub_qa_pairs: list[tuple[str, str]],
    target_entities: list[str],
) -> list[str]:
    """
    Phase 1 seulement.
    Pour chaque entité cible, vérifie si TOUTES les SQ la concernant
    retournent source_type='aucune_donnee' ou valeur=null.
    sub_qa_pairs : [(sub_question, json_str), ...]
    """
    if not target_entities:
        return []

    silence = []
    for entity in target_entities:
        e_low  = entity.lower()
        e_norm = _nfkd(e_low)

        # SQ qui mentionnent explicitement cette entité
        entity_sqs = [
            (sq, js) for sq, js in sub_qa_pairs
            if e_low in sq.lower() or e_norm in _nfkd(sq.lower())
        ]

        if entity_sqs:
            # Entité nommée dans au moins une SQ → vérifier si toutes sont N/D
            if all(_json_is_no_data(js) for _, js in entity_sqs):
                silence.append(entity)
        elif all(_json_is_no_data(js) for _, js in sub_qa_pairs):
            # Entité non nommée dans les SQ mais toutes les SQ sont N/D
            # (cas d'une question ciblée sur une unique commune)
            silence.append(entity)

    return silence


def build_silence_list_baseline(
    target_entities: list[str],
    sources: list[dict],
) -> list[str]:
    """
    Baseline.
    Entité silencée si aucune source qualitative (verbatim ou enquête-commune)
    ne la mentionne.
    """
    silence = []
    for entity in target_entities:
        e_low = _nfkd(entity.lower().strip())
        qualitative = [
            s for s in sources
            if _nfkd(str(s.get("commune", "")).lower().strip()) == e_low
            and (
                s.get("type") is None           # verbatim
                or s.get("view") == "enquete_commune"
            )
        ]
        if not qualitative:
            silence.append(entity)
    return silence


def _json_is_no_data(json_str: str) -> bool:
    try:
        obj = json.loads(json_str)
        return (
            obj.get("source_type") == "aucune_donnee"
            or obj.get("valeur") is None
        )
    except Exception:
        return False


def check_entity_silence(
    answer: str,
    silence_list: list[str],
    quali_silence: list[str] | None = None,
) -> list[Violation]:
    """
    Vérification par blocs structurés [[VERDICT]] / [[CONTEXTE_ELARGI]].

    Règles :
      VERDICT        — seuls les constats d'absence sont autorisés pour les entités silencées.
                       Toute mention sans marqueur d'absence → violation.
      CONTEXTE_ELARGI — la mention d'une entité silencée est totalement interdite
                        (les données viennent d'une autre scope).
      Hors blocs (outer) — aucun énoncé sur une entité silencée, même avec marqueur.

    Rétrocompatibilité : si aucun bloc détecté, applique les anciennes règles
    sur le texte complet (hors headers markdown).
    """
    if not silence_list and not quali_silence:
        return []

    _HEADER_RE = re.compile(r"^\s*#{1,6}\s|^\s*\*{1,3}[^*]+\*{1,3}\s*$|^[-=]{3,}\s*$")

    def _sentences(text: str) -> list[str]:
        return [s for s in re.split(r"(?<=[.!?\n])\s+", text)
                if not _HEADER_RE.match(s.strip()) and s.strip()]

    blocks = _parse_blocks(answer)
    has_blocks = bool(blocks["verdict"] or blocks["contexte_elargi"])

    violations: list[Violation] = []
    all_entities = list({*(silence_list or []), *(quali_silence or [])})

    def _mentions(sent: str, entity: str) -> bool:
        e_low = entity.lower()
        return e_low in sent.lower() or _nfkd(e_low) in _nfkd(sent.lower())

    if has_blocks:
        # ── Zone VERDICT : absence-markers autorisés, énoncés factuels interdits ──
        for sent in _sentences(blocks["verdict"]):
            for entity in (silence_list or []):
                if _mentions(sent, entity) and not _ABSENCE_RE.search(sent):
                    violations.append(Violation(
                        rule="silence",
                        detail=f"[VERDICT] Entité silencée '{entity}' sans marqueur d'absence",
                        evidence=sent.strip()[:200],
                        context=f"Bloc VERDICT : seuls les constats d'absence sont autorisés pour '{entity}'",
                    ))
            for entity in (quali_silence or []):
                if _mentions(sent, entity) and not _ABSENCE_RE.search(sent):
                    if _QUALI_LANG_RE.search(sent):
                        violations.append(Violation(
                            rule="silence",
                            detail=f"[VERDICT] Entité '{entity}' : énoncé qualitatif sans verbatims",
                            evidence=sent.strip()[:200],
                            context="Bloc VERDICT : scores uniquement ou constat d'absence",
                        ))

        # ── Zone CONTEXTE_ELARGI : mention de l'entité silencée interdite ──
        for sent in _sentences(blocks["contexte_elargi"]):
            for entity in all_entities:
                if _mentions(sent, entity):
                    violations.append(Violation(
                        rule="silence",
                        detail=f"[CONTEXTE_ELARGI] Entité silencée '{entity}' citée (scope différent)",
                        evidence=sent.strip()[:200],
                        context="Bloc CONTEXTE_ELARGI : données d'une autre scope — ne pas nommer l'entité cible",
                    ))

        # ── Zone OUTER : aucun énoncé sur entité silencée ──
        for sent in _sentences(blocks["outer"]):
            for entity in all_entities:
                if _mentions(sent, entity):
                    violations.append(Violation(
                        rule="silence",
                        detail=f"[hors bloc] Entité '{entity}' mentionnée en dehors des blocs structurés",
                        evidence=sent.strip()[:200],
                        context="Hors [[VERDICT]]/[[CONTEXTE_ELARGI]] : aucun énoncé autorisé sur l'entité",
                    ))

    else:
        # Balises absentes → violation structurelle bruyante (pas de repli silencieux).
        # En mode silence, le synthétiseur est contraint par grammaire GBNF.
        # Si les balises manquent quand même, c'est un bug pipeline, pas un cas normal.
        violations.append(Violation(
            rule="silence",
            detail="[STRUCTURE] Balises [[VERDICT]]...[[/VERDICT]] absentes",
            evidence=answer[:120].strip(),
            context="En mode silence, la réponse DOIT être encadrée par [[VERDICT]]...[[/VERDICT]] "
                    "(grammaire GBNF + instruction prompt). Absence = défaillance structurelle.",
        ))

    return violations


# ── Règle 2 : number grounding ───────────────────────────────────────────────

def check_numbers(
    answer: str,
    synth_context: str,
) -> list[Violation]:
    """
    Chaque valeur numérique significative de la réponse doit figurer
    (forme normalisée) dans synth_context (contexte injecté au synthétiseur).
    """
    violations = []

    # Construire l'ensemble des nombres présents dans le contexte
    ctx_nums: set[str] = set()
    for m in _NUM_RE.finditer(synth_context):
        raw = m.group(1)
        ctx_nums.add(raw)
        ctx_nums.add(_normalize_num(raw))

    seen: set[str] = set()
    for m in _NUM_RE.finditer(answer):
        raw = m.group(1)
        if raw in seen:
            continue
        seen.add(raw)

        if _is_trivial_int(raw) or _is_year(raw):
            continue

        norm = _normalize_num(raw)
        if raw not in ctx_nums and norm not in ctx_nums:
            # Vérification substring (pour les formes composites comme "6,8/10")
            if raw not in synth_context and norm not in synth_context:
                violations.append(Violation(
                    rule="number",
                    detail=f"Valeur '{raw}' absente du contexte de synthèse",
                    evidence=raw,
                    context="Nombre introuvable dans les réponses SQ injectées",
                ))
    return violations


# ── Règle 3 : quote grounding ────────────────────────────────────────────────

def check_quotes(
    answer: str,
    sources: list[dict],
) -> list[Violation]:
    """
    Toute citation entre guillemets > 5 mots doit se retrouver
    littéralement dans les extraits des sources récupérées.
    """
    violations = []
    all_src_text = " ".join(
        str(s.get("extrait", s.get("content", ""))) for s in sources
    ).lower()

    for m in _QUOTE_RE.finditer(answer):
        q = m.group(1).strip()
        if len(q.split()) > 5:
            if q.lower() not in all_src_text:
                violations.append(Violation(
                    rule="quote",
                    detail="Citation >5 mots introuvable dans les sources",
                    evidence=q[:150],
                    context="Aucune correspondance littérale dans les extraits",
                ))
    return violations


# ── Runner principal ─────────────────────────────────────────────────────────

class HallucinationGuard:

    def run_all(
        self,
        final_answer: str,
        silence_list: list[str],
        synth_context: str,
        sources: list[dict],
        silence_map: dict | None = None,
    ) -> GuardReport:
        # Dériver quali_silence depuis silence_map (quanti=True, quali=False)
        quali_silence: list[str] = []
        if silence_map:
            quali_silence = [
                c for c, p in silence_map.items()
                if not p.get("quali") and p.get("quanti")
            ]

        v_silence = check_entity_silence(final_answer, silence_list, quali_silence)
        v_numbers = check_numbers(final_answer, synth_context)
        v_quotes  = check_quotes(final_answer, sources)
        return GuardReport(
            silence_list=silence_list,
            silence_map=silence_map or {},
            violations=v_silence + v_numbers + v_quotes,
        )

    def print_report(self, report: GuardReport, prefix: str = "") -> None:
        tag = f"{prefix}[GUARD]"
        if report.passed:
            print(f"{tag} ✓  Aucune violation — réponse fondée")
            if report.silence_list:
                print(f"{tag}    liste_silence={report.silence_list} (aucun énoncé non fondé)")
            if report.silence_map:
                for commune, presence in report.silence_map.items():
                    if not presence.get("quali") and presence.get("quanti"):
                        print(f"{tag}    {commune} — quanti présent, quali absent (verbatims)")
            return

        bar = "━" * 65
        print(f"\n{bar}")
        print(f"{tag} ⚠  {len(report.violations)} VIOLATION(S) DÉTECTÉE(S)")
        if report.silence_list:
            print(f"  liste_silence (total) : {', '.join(report.silence_list)}")
        if report.silence_map:
            for commune, presence in report.silence_map.items():
                if not presence.get("quali") and presence.get("quanti"):
                    print(f"  liste_silence (quali) : {commune} — verbatims absents, scores disponibles")
        for i, v in enumerate(report.violations, 1):
            label = {"silence": "SILENCE", "number": "NOMBRE", "quote": "CITATION"}[v.rule]
            print(f"  [{i}] {label} — {v.detail}")
            print(f"       Evidence : {v.evidence[:120]}")
            if v.context:
                print(f"       Attendu  : {v.context[:120]}")
        print(f"{bar}\n")


# ── Helpers pour injection dans la question ──────────────────────────────────

def build_silence_header(silence_list: list[str]) -> str:
    """Retourne le texte à préfixer à initial_question pour le synthétiseur."""
    if not silence_list:
        return ""
    entities_str = ", ".join(f"'{e}'" for e in silence_list)
    return (
        f"[INSTRUCTION SYNTHÈSE — ENTITÉS SANS DONNÉES]\n"
        f"Les entités suivantes n'ont AUCUNE donnée vérifiée dans le corpus : {entities_str}.\n"
        f"Règle absolue : ne formuler aucun énoncé factuel, chiffré ou qualitatif sur ces entités. "
        f"Si elles doivent être mentionnées, utiliser uniquement une formulation d'absence explicite "
        f"(ex. : 'aucune donnée disponible pour X', 'les informations sur X sont absentes du corpus').\n\n"
    )


def build_silence_map_from_chroma(
    communes: list[str],
    chroma_path: str = "./chroma_portrait",
) -> dict[str, dict[str, bool]]:
    """
    Option B — vérification directe dans ChromaDB, indépendamment du LLM.

    Pour chaque commune, interroge les collections de référence :
      - portrait_verbatims       → quali (verbatims citoyens, champ "nom")
      - enquete_scores_commune   → quanti/enquête (champ "commune")
      - oppchovec_scores         → quanti/OppChoVec (360 communes, champ "commune")

    quanti=True si la commune apparaît dans enquete_scores_commune OU oppchovec_scores.

    Retourne {commune: {"quali": bool, "quanti": bool}}.
    Communes présentes dans les deux registres sont exclues du résultat (rien à silencer).
    """
    if not communes:
        return {}

    try:
        import chromadb as _cdb
        client = _cdb.PersistentClient(path=chroma_path)
    except Exception:
        return {}

    try:
        verbatims_col = client.get_collection("portrait_verbatims")
    except Exception:
        verbatims_col = None

    try:
        scores_col = client.get_collection("enquete_scores_commune")
    except Exception:
        scores_col = None

    try:
        oppchovec_col = client.get_collection("oppchovec_scores")
    except Exception:
        oppchovec_col = None

    result: dict[str, dict[str, bool]] = {}
    for commune in communes:
        quali = False
        quanti = False

        if verbatims_col is not None:
            try:
                # portrait_verbatims utilise le champ "nom" (pas "commune")
                res = verbatims_col.get(
                    where={"nom": {"$eq": commune}},
                    include=["metadatas"],
                    limit=1,
                )
                quali = bool(res["metadatas"])
            except Exception:
                pass

        if not quanti and scores_col is not None:
            try:
                res = scores_col.get(
                    where={"commune": {"$eq": commune}},
                    include=["metadatas"],
                    limit=1,
                )
                quanti = bool(res["metadatas"])
            except Exception:
                pass

        if not quanti and oppchovec_col is not None:
            try:
                res = oppchovec_col.get(
                    where={"commune": {"$eq": commune}},
                    include=["metadatas"],
                    limit=1,
                )
                quanti = bool(res["metadatas"])
            except Exception:
                pass

        # N'inclure que les communes partiellement ou totalement absentes
        if not quali or not quanti:
            result[commune] = {"quali": quali, "quanti": quanti}

    return result


def build_silence_header_with_map(silence_map: dict[str, dict[str, bool]]) -> str:
    """
    Injection pré-synthèse avec format par blocs [[VERDICT]] / [[CONTEXTE_ELARGI]].

    Le synthétiseur doit encadrer sa réponse dans des blocs structurés :
      [[VERDICT]] réponse principale [[/VERDICT]]
      [[CONTEXTE_ELARGI scope="..."]] données contextuelles [[/CONTEXTE_ELARGI]]  (optionnel)

    Les balises sont retirées avant affichage.
    """
    if not silence_map:
        return ""

    full_silence = [c for c, p in silence_map.items() if not p["quali"] and not p["quanti"]]
    quali_only   = [c for c, p in silence_map.items() if not p["quali"] and p["quanti"]]

    parts = [
        "[INSTRUCTION SYNTHÈSE — FORMAT DE RÉPONSE STRUCTURÉ]\n"
        "Ta réponse DOIT commencer par [[VERDICT]] et se terminer par [[/VERDICT]].\n"
        "Si des données contextuelles d'une autre échelle sont utiles, ajoute ensuite "
        "[[CONTEXTE_ELARGI scope=\"corse_entiere\"]]...[[/CONTEXTE_ELARGI]].\n"
        "Ne place aucun contenu en dehors de ces balises.\n\n"
    ]

    if full_silence:
        ents = ", ".join(f"'{e}'" for e in full_silence)
        parts.append(
            f"ENTITÉS SANS AUCUNE DONNÉE : {ents}.\n"
            f"Dans [[VERDICT]] : mentionne uniquement l'absence "
            f"('aucune donnée disponible pour X dans le corpus').\n"
            f"Dans [[CONTEXTE_ELARGI]] : n'utilise PAS le nom de ces entités — "
            f"les données sont d'une autre scope.\n"
        )

    if quali_only:
        ents = ", ".join(f"'{e}'" for e in quali_only)
        parts.append(
            f"ENTITÉS AVEC SCORES UNIQUEMENT (verbatims absents) : {ents}.\n"
            f"Dans [[VERDICT]] : cite les scores OppChoVec (indicateurs chiffrés). "
            f"Interdit : énoncés qualitatifs sur le ressenti des habitants. "
            f"Conclus explicitement que les verbatims sont absents.\n"
            f"Dans [[CONTEXTE_ELARGI]] : n'utilise PAS le nom de ces entités.\n"
        )

    parts.append("\n")
    return "".join(parts)
