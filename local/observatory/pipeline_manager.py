"""
observatory/pipeline_manager.py
================================
Thin, fully reversible wrapper around the RAPTOR v10 pipeline
(``rag_v10_raptor_subq``) for the *Dumè GPT* well-being observation assistant.

Design constraints
------------------
* The RAG source files are **never edited**. Model overrides and progress
  instrumentation are installed as monkey-patched shims on the ``rag_v10``
  module, once, at import time.
* Queries are **serialised** (the pipeline relies on module-global state).
* Coarse progress events are emitted per stage: ``decompose`` ->
  ``answer`` (i of n) -> ``synthesize`` -> ``score``.
* A trusted caller (dev console) may override, per query: the model behind each
  of the three LLM roles (decomposer / answerer / synthesizer), independently,
  across Mistral / Claude / OpenAI; plus ``k``, ``n_subquestions``, temperature,
  pipeline version and output language. Ordinary visitors get ``DEFAULTS`` (a
  low-cost all-working config); ``REFERENCE_PIPELINE`` is the documented thesis
  config (Mistral Large x2 + Claude Haiku), one click away in the console.

Public surface
--------------
``start_loading()``            begin loading models in a background thread
``status()``                   -> {"ready", "error", "v11"}
``run(question, cfg, emit)``   -> result dict  (raises ``Busy`` / ``NotReady``)
``DEFAULTS`` / ``REFERENCE_PIPELINE`` / ``PROVIDER_MODELS`` / ``ROLES`` / ``VERSION_CHOICES``
``commune_names()`` / ``commune_profile(name)``
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Repo wiring — chroma paths in the RAG code are relative to the repo root.
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    os.chdir(_REPO_ROOT)
except OSError:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

import rag_v10_raptor_subq as _v10  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration surface
# --------------------------------------------------------------------------- #
# The v10 pipeline uses three LLM roles, each independently selectable here:
#   decomposer  — splits the question into sub-questions        (v10: Mistral Large)
#   answerer    — answers each sub-question from retrieved text  (v10: Claude Haiku)
#   synthesizer — writes the final answer + dimension score      (v10: Mistral Large)
ROLES = ["decomposer", "answerer", "synthesizer"]

PROVIDER_MODELS = {
    "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
    "claude": ["claude-haiku-4-5-20251001", "claude-haiku-4-5", "claude-sonnet-4-6",
               "claude-sonnet-5", "claude-opus-5"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
}

OUTPUT_LANGUAGES = ["en", "fr"]

# What everyone gets unless the console overrides it: low-cost, all keys working.
DEFAULTS = {
    "version": "v10",
    "decomposer": {"provider": "openai", "model": "gpt-4o-mini"},
    "answerer": {"provider": "claude", "model": _v10.ANSWERER_MODEL},
    "synthesizer": {"provider": "openai", "model": "gpt-4o"},
    "k": 5,
    "n_subquestions": _v10.DEFAULT_N_SUBQUESTIONS,
    "temperature": None,                       # None => the pipeline's tuned temps
    "commune": "",                             # "" => all of Corsica
    "output_language": "fr",                   # "en" | "fr" — French is the base language;
                                                # English is a hidden console-only switch (article screenshots)
}

# The exact configuration used for every evaluation run in the thesis so far.
# Exposed as a one-click preset in the console. Restore it to reproduce the
# documented pipeline (needs a working Mistral subscription).
REFERENCE_PIPELINE = {
    "version": "v10",
    "decomposer": {"provider": "mistral", "model": _v10.DECOMPOSER_MODEL},     # mistral-large-latest
    "answerer": {"provider": "claude", "model": _v10.ANSWERER_MODEL},          # claude-haiku-4-5-20251001
    "synthesizer": {"provider": "mistral", "model": _v10.SYNTHESIZER_MODEL},   # mistral-large-latest
    "k": 5,
    "n_subquestions": _v10.DEFAULT_N_SUBQUESTIONS,
    "temperature": None,
}

# Appended to every LLM system prompt when output_language == "en".
# The RAG source prompts are written in French; this steers the visible output
# to English without touching those files.
_EN_SUFFIX = (
    "\n\nOUTPUT LANGUAGE: Write ALL of your output in English — every heading, "
    "sentence, list item and JSON string value. Translate any French source "
    "material you quote or paraphrase. Keep proper nouns (commune names, "
    "institutions, indicator names such as OppChoVec) unchanged."
)

VERSION_CHOICES = ["v10"]  # "v11" appended at load time if it initialises


class Busy(RuntimeError):
    """Raised when a query is already in flight."""


class NotReady(RuntimeError):
    """Raised before the pipeline has finished loading."""


# --------------------------------------------------------------------------- #
# Captured originals (before any patching)
# --------------------------------------------------------------------------- #
_ORIG_CALL_CLAUDE = _v10._call_claude
_ORIG_CALL_MISTRAL = _v10._call_mistral
_ORIG_DECOMPOSE = _v10.decompose_question
_ORIG_ANSWER = _v10.answer_subquestion
_ORIG_SYNTHESIZE = _v10.synthesize_answers
_ORIG_SCORE = _v10.score_dimension

_LOCK = threading.Lock()
_CUR: Optional[dict] = None          # active-run context, guarded by _LOCK

_READY = False
_LOAD_ERR: Optional[str] = None
_V10 = None
_V11 = None

_COMMUNE_STATS: dict = {}            # normalized name -> survey (subjective) profile
_COMMUNE_MEANS: dict = {}            # survey metric -> Corsica-wide mean
_COMMUNE_OBJ: dict = {}             # normalized name -> OppChoVec (objective) profile


# --------------------------------------------------------------------------- #
# Instrumentation helpers (run inside the query worker thread — and, with
# parallel sub-questions, inside several answerer threads concurrently, hence
# the lock guarding every _CUR mutation below).
# --------------------------------------------------------------------------- #
_CUR_LOCK = threading.Lock()


def _emit(stage: str, **extra) -> None:
    cur = _CUR
    if cur and cur.get("emit"):
        try:
            cur["emit"]({"stage": stage, **extra})
        except Exception:
            pass


def _accum(step: str, dt: float, ptok: int, ctok: int) -> None:
    cur = _CUR
    if not cur:
        return
    with _CUR_LOCK:
        s = cur["steps"].setdefault(
            step, {"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
        )
        s["elapsed_s"] += dt
        s["prompt_tokens"] += ptok
        s["completion_tokens"] += ctok
        s["n_calls"] += 1


def _drop_temperature(model: str) -> bool:
    """5-family models reject sampling params (temperature/top_p)."""
    return any(tag in model for tag in ("sonnet-5", "opus-5", "fable"))


# --------------------------------------------------------------------------- #
# LLM call shims — override the model / provider, capture token usage & timing.
# Retry/backoff mirrors the originals in rag_v10_raptor_subq.
# --------------------------------------------------------------------------- #
def _anthropic_chat(model, system_prompt, prompt, max_tokens, temperature, max_retries):
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) is not set")
    client = anthropic.Anthropic(api_key=api_key, timeout=300.0)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not _drop_temperature(model):
        kwargs["temperature"] = temperature

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            usage = getattr(resp, "usage", None)
            text = next((b.text for b in resp.content
                         if getattr(b, "type", None) == "text"), "")
            return text, (time.time() - t0), \
                int(getattr(usage, "input_tokens", 0) or 0), \
                int(getattr(usage, "output_tokens", 0) or 0)
        except Exception as exc:                                   # noqa: BLE001
            err = str(exc).lower()
            retryable = any(x in err for x in
                            ("429", "529", "overload", "rate", "timeout",
                             "timed out", "connection"))
            if retryable and attempt < max_retries - 1:
                time.sleep(2 ** attempt * 4)
                continue
            raise


def _openai_chat(base_url, api_key, model, system_prompt, prompt,
                 max_tokens, temperature, max_retries):
    from openai import OpenAI

    if not api_key:
        raise RuntimeError("API key missing for OpenAI-compatible endpoint")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = getattr(resp, "usage", None)
            return resp.choices[0].message.content, (time.time() - t0), \
                int(getattr(usage, "prompt_tokens", 0) or 0), \
                int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception as exc:                                   # noqa: BLE001
            err = str(exc).lower()
            if ("429" in err or "rate" in err) and attempt < max_retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise


def _lang_system(system_prompt: str) -> str:
    cur = _CUR
    if cur and cur.get("lang") == "en":
        return (system_prompt or "") + _EN_SUFFIX
    return system_prompt


def _dispatch(role_cfg, system_prompt, prompt, max_tokens, temperature, max_retries):
    """Route one LLM call to the provider chosen for this role."""
    provider = role_cfg["provider"]
    model = role_cfg["model"]
    sysp = _lang_system(system_prompt)
    if provider == "claude":
        return _anthropic_chat(model, sysp, prompt, max_tokens, temperature, max_retries)
    if provider == "openai":
        return _openai_chat(None, os.getenv("OPENAI_API_KEY"), model, sysp, prompt,
                            max_tokens, temperature, max_retries)
    return _openai_chat(_v10.MISTRAL_BASE_URL, os.getenv("MISTRAL_API_KEY"), model,
                        sysp, prompt, max_tokens, temperature, max_retries)


def _role_cfg(role: str) -> dict:
    cur = _CUR or {}
    return cur.get(role) or DEFAULTS[role]


# --------------------------------------------------------------------------- #
# Conversation memory — condense a follow-up into a standalone question.
# The pipeline itself (rag_v10_raptor_subq.py) never sees the conversation;
# this runs once, before query(), and hands v10 a single plain-English
# question exactly as if it were asked fresh. No RAG source file is touched.
# --------------------------------------------------------------------------- #
_CONDENSE_SYSTEM = (
    "You rewrite a follow-up question into a standalone question, using the "
    "recent conversation for context. Resolve pronouns and implicit "
    "references (\"what about Bastia?\", \"and the other one?\", \"why is "
    "that?\") into explicit ones — commune names, indicators, topics — drawn "
    "from the conversation below. If the follow-up is already standalone, "
    "return it unchanged, verbatim. Reply with ONLY the rewritten question — "
    "no preamble, no quotes, no explanation."
)
_MAX_HISTORY_TURNS = 3       # exchanges (user+bot pairs), not raw entries
_MAX_ANSWER_CHARS = 350      # per prior answer, in the condense prompt only


def _condense_question(question: str, history: list) -> str:
    """Best-effort: fold `history` + `question` into one standalone question
    via the decomposer role. Falls back to `question` unchanged on any error
    or empty history — never blocks the real answer over this optional step."""
    turns = [h for h in (history or [])
             if isinstance(h, dict) and h.get("role") in ("user", "bot") and h.get("text")]
    turns = turns[-(_MAX_HISTORY_TURNS * 2):]
    if not turns:
        return question

    lines = []
    for h in turns:
        text = str(h["text"])
        if h["role"] == "bot" and len(text) > _MAX_ANSWER_CHARS:
            text = text[:_MAX_ANSWER_CHARS] + "…"
        lines.append(("Q: " if h["role"] == "user" else "A: ") + text)
    prompt = "\n".join(lines) + f"\n\nFollow-up: {question}"

    cur = _CUR
    if cur is not None:
        cur["stage"] = "condense"
    _emit("condense")
    try:
        text, dt, ptok, ctok = _dispatch(
            _role_cfg("decomposer"), _CONDENSE_SYSTEM, prompt,
            max_tokens=200, temperature=0.1, max_retries=3)
        _accum("condense", dt, ptok, ctok)
        rewritten = (text or "").strip().strip('"')
        return rewritten or question
    except Exception as exc:                                        # noqa: BLE001
        print(f"[obs] condense failed, using original question: {exc}")
        return question


def _call_claude_shim(prompt, system_prompt, model=None, max_tokens=800,
                      temperature=0.2, max_retries=5, **_):
    """Answerer step (v10 routes it through _call_claude). Provider/model per console."""
    cur = _CUR
    text, dt, ptok, ctok = _dispatch(
        _role_cfg("answerer"), system_prompt, prompt, max_tokens, temperature, max_retries)
    _accum((cur or {}).get("stage", "answer"), dt, ptok, ctok)
    return text


def _call_mistral_shim(prompt, system_prompt, model=None, max_tokens=1000,
                       temperature=0.3, max_retries=5, **_):
    """Decompose / synthesize / score (v10 routes these through _call_mistral).
    decompose -> 'decomposer' role;  synthesize & score -> 'synthesizer' role."""
    cur = _CUR
    stage = (cur or {}).get("stage", "decompose")
    role = "decomposer" if stage == "decompose" else "synthesizer"
    text, dt, ptok, ctok = _dispatch(
        _role_cfg(role), system_prompt, prompt, max_tokens, temperature, max_retries)
    _accum(stage, dt, ptok, ctok)
    return text


# --------------------------------------------------------------------------- #
# Stage shims — set the current step label and push a progress event.
# --------------------------------------------------------------------------- #
def _decompose_shim(*a, **kw):
    if _CUR is not None:
        _CUR["stage"] = "decompose"
    _emit("decompose")
    result = _ORIG_DECOMPOSE(*a, **kw)
    try:
        if _CUR is not None:
            _CUR["n_sub"] = len(result)
            _emit("decomposed", n=len(result))
    except TypeError:
        pass
    return result


def _answer_shim(*a, **kw):
    # May run concurrently across sub-questions (parallel path) — lock the
    # counter so "i" is a clean 1..n sequence of *completions*, not a race.
    if _CUR is not None:
        _CUR["stage"] = "answer"
        with _CUR_LOCK:
            _CUR["answer_i"] = _CUR.get("answer_i", 0) + 1
            i = _CUR["answer_i"]
        _emit("answer", i=i, n=_CUR.get("n_sub", 0))
    return _ORIG_ANSWER(*a, **kw)


def _synthesize_shim(*a, **kw):
    if _CUR is not None:
        _CUR["stage"] = "synthesize"
    _emit("synthesize")
    return _ORIG_SYNTHESIZE(*a, **kw)


def _score_shim(*a, **kw):
    if _CUR is not None:
        _CUR["stage"] = "score"
    _emit("score")
    return _ORIG_SCORE(*a, **kw)


def _install_patches() -> None:
    _v10._call_claude = _call_claude_shim
    _v10._call_mistral = _call_mistral_shim
    _v10.decompose_question = _decompose_shim
    _v10.answer_subquestion = _answer_shim
    _v10.synthesize_answers = _synthesize_shim
    _v10.score_dimension = _score_shim


# --------------------------------------------------------------------------- #
# "Bien-etre global" pre-fetch — ported from api_server_multi_version.py so the
# v10 decomposer gets the same guidance it does in the reference server.
# --------------------------------------------------------------------------- #
_BIENEETRE_KEYWORDS = [
    "bien-être", "bien être", "bienetre", "bien-etre", "qualité de vie",
    "qualite de vie", "comment se porte", "comment vivent", "comment vit",
    "portrait global", "portrait de", "satisfaction globale", "conditions de vie",
    "well-being", "wellbeing", "quality of life",
]
_QUALI_OVERRIDE_KEYWORDS = [
    "ressenti", "ressentent", "ressent", "ressens", "perçoivent", "perçoit",
    "perception", "perceptions", "que pensent", "que pense", "avis des", "opinion",
    "témoignages", "verbatims", "verbatim", "enquête citoyenne", "enquete citoyenne",
    "comment vivent-ils", "comment vivent-elles", "comment les habitants",
    "comment les résidents", "comment les gens", "how do residents",
    "how do inhabitants", "perceive", "describe",
]


def _is_bieneetre_question(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in _QUALI_OVERRIDE_KEYWORDS):
        return False
    return any(kw in q for kw in _BIENEETRE_KEYWORDS)


def _v10_extra_context(rag, question: str) -> dict:
    bieneetre = _is_bieneetre_question(question)
    ctx = ""
    if bieneetre:
        try:
            col = rag.retriever._extra_cols.get("enquete_scores_commune")
            if col is not None:
                q_emb = rag.retriever._encode_query(question)
                res = col.query(
                    query_embeddings=[q_emb],
                    n_results=min(3, col.count()),
                    include=["documents", "metadatas"],
                )
                docs = res["documents"][0] if res.get("documents") else []
                if docs:
                    ctx = ("[Scores enquête par commune (subjectif/quanti)]\n"
                           + "\n\n".join(docs))
        except Exception:
            ctx = ""
    return {"ctx": ctx, "bieneetre": bieneetre}


# --------------------------------------------------------------------------- #
# Commune profiles for the sidebar (from df_mean_by_commune.csv)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


_HEADLINE = [
    ("Score bonheur", "Happiness", 5.0),
    ("Score qualité de vie", "Quality of life", 5.0),
    ("Score confiance avenir", "Confidence in the future", 5.0),
]
_DIMENSIONS = [
    "Transports", "Éducation", "Santé", "Logement", "Revenus", "Sécurité",
    "Institutions", "Services locaux", "Culture", "Soutien social",
    "Vie associative", "Situation pro", "Temps travail/perso",
]


def _load_commune_stats() -> None:
    global _COMMUNE_STATS, _COMMUNE_MEANS
    path = os.path.join(_REPO_ROOT, "df_mean_by_commune.csv")
    if not os.path.exists(path):
        print("[obs] df_mean_by_commune.csv not found — sidebar indicators disabled")
        return
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    def fnum(row, key):
        try:
            return float(str(row.get(key, "")).replace(",", "."))
        except (TypeError, ValueError):
            return None

    sums, counts = {}, {}
    for row in rows:
        name = (row.get("commune") or "").strip()
        if not name:
            continue
        prof = {
            "name": name,
            "respondents": int(fnum(row, "total_respondants") or 0),
            "mean_age": fnum(row, "Age"),
            "headline": {},
            "dimensions": {},
        }
        for col, _label, _mx in _HEADLINE:
            v = fnum(row, col)
            if v is not None:
                prof["headline"][col] = v
                sums[col] = sums.get(col, 0.0) + v
                counts[col] = counts.get(col, 0) + 1
        for col in _DIMENSIONS:
            v = fnum(row, col)
            if v is not None:
                prof["dimensions"][col] = v
        _COMMUNE_STATS[_norm(name)] = prof

    _COMMUNE_MEANS = {c: (sums[c] / counts[c]) for c in sums if counts.get(c)}
    print(f"[obs] commune profiles loaded: {len(_COMMUNE_STATS)}")


def _load_objective_stats() -> None:
    """OppChoVec territorial scores for all 360 Corsican communes, read once from
    the already-open `oppchovec_scores` ChromaDB collection on the v10 retriever."""
    global _COMMUNE_OBJ
    try:
        col = getattr(_V10.retriever, "_oppchovec", None)
        if col is None:
            print("[obs] oppchovec_scores collection not available — objective view disabled")
            return
        res = col.get(where={"source": "oppchovec_betti_0_10"}, include=["metadatas"])
        for m in res.get("metadatas", []):
            name = (m.get("commune") or "").strip()
            if not name:
                continue
            _COMMUNE_OBJ[_norm(name)] = {
                "name": name,
                "oppchovec": m.get("oppchovec_0_10"),
                "opp": m.get("opp_0_10"), "cho": m.get("cho_0_10"), "vec": m.get("vec_0_10"),
                "rank_total": m.get("rank_total"), "rank_opp": m.get("rank_opp"),
                "rank_cho": m.get("rank_cho"), "rank_vec": m.get("rank_vec"),
            }
        print(f"[obs] objective (OppChoVec) profiles loaded: {len(_COMMUNE_OBJ)}")
    except Exception as exc:                                       # noqa: BLE001
        print(f"[obs] objective stats load failed: {exc}")


def commune_names() -> list:
    names = {p["name"] for p in _COMMUNE_OBJ.values()}
    names |= {p["name"] for p in _COMMUNE_STATS.values()}
    return sorted(names, key=str.lower)


_N_COMMUNES = 360


def _rank_tone(rank):
    if not rank:
        return "—", "flat"
    if rank <= _N_COMMUNES / 3:
        return "▲", "good"
    if rank >= 2 * _N_COMMUNES / 3:
        return "▼", "bad"
    return "—", "flat"


def _profile_objective(name: str) -> dict:
    o = _COMMUNE_OBJ.get(_norm(name or ""))
    if not o:
        return {"name": name, "view": "objective", "available": False}
    rows = [
        ("OppChoVec (overall)", o["oppchovec"], o["rank_total"]),
        ("Opportunities (Opp)", o["opp"], o["rank_opp"]),
        ("Choice (Cho)", o["cho"], o["rank_cho"]),
        ("Lived experience (Vec)", o["vec"], o["rank_vec"]),
    ]
    indicators = []
    for label, val, rank in rows:
        if val is None:
            continue
        arrow, tone = _rank_tone(rank)
        indicators.append({
            "label": label,
            "value": round(float(val), 1),
            "max": 10,
            "rank": rank,
            "n": _N_COMMUNES,
            "arrow": arrow,
            "tone": tone,
        })
    return {
        "name": o["name"],
        "view": "objective",
        "available": True,
        "rank_total": o["rank_total"],
        "note": "OppChoVec territorial scores — all 360 Corsican communes.",
        "indicators": indicators,
    }


def _profile_subjective(name: str) -> dict:
    prof = _COMMUNE_STATS.get(_norm(name or ""))
    if not prof:
        return {"name": name, "view": "subjective", "available": False}
    indicators = []
    for col, label, mx in _HEADLINE:
        val = prof["headline"].get(col)
        if val is None:
            continue
        mean = _COMMUNE_MEANS.get(col)
        delta = None if mean is None else round(val - mean, 2)
        if delta is None or abs(delta) < 0.15:
            arrow, tone = "—", "flat"
        elif delta > 0:
            arrow, tone = "▲", "good"
        else:
            arrow, tone = "▼", "bad"
        indicators.append({
            "label": label, "value": round(val, 1), "max": mx,
            "vs_corsica": delta, "arrow": arrow, "tone": tone,
        })
    dims = prof["dimensions"]
    best = max(dims, key=dims.get) if dims else None
    worst = min(dims, key=dims.get) if dims else None
    return {
        "name": prof["name"],
        "view": "subjective",
        "available": True,
        "respondents": prof["respondents"],
        "mean_age": round(prof["mean_age"], 1) if prof["mean_age"] else None,
        "best_dimension": best,
        "worst_dimension": worst,
        "note": "Citizen-survey perceptions — 68 communes with responses.",
        "indicators": indicators,
    }


def commune_profile(name: str, view: str = "objective") -> dict:
    if view == "subjective":
        return _profile_subjective(name)
    return _profile_objective(name)


# --------------------------------------------------------------------------- #
# Result shaping
# --------------------------------------------------------------------------- #
_SQ_MARKER_RE = re.compile(r"\s*\[\s*/?(?:SQ\s*\d+(?:\s*[,;]\s*SQ\s*\d+)*|SOURCES?_MOBILIS[EÉ]ES?)\s*\]", re.IGNORECASE)


def _polish_answer(text: str) -> str:
    """Strip the pipeline's internal inline citation markers ([SQ1], [SQ2, SQ3], …)
    that are meant for evaluation, not for a decision-maker reading the answer."""
    if not text:
        return text
    out = _SQ_MARKER_RE.sub("", text)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _clean_sources(sources) -> list:
    out = []
    for s in (sources or [])[:30]:
        if not isinstance(s, dict):
            out.append({"extrait": str(s)[:800]})
            continue
        clean = {}
        for k, v in s.items():
            if k in ("embedding", "vector"):
                continue
            if isinstance(v, str) and len(v) > 800:
                v = v[:800] + "…"
            try:
                json.dumps(v)
                clean[k] = v
            except TypeError:
                clean[k] = str(v)[:800]
        out.append(clean)
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _force_cpu_embeddings() -> None:
    try:
        import sentence_transformers as _st
    except Exception as exc:                                       # noqa: BLE001
        print(f"[obs] sentence_transformers import failed: {exc}")
        return
    _orig_init = _st.SentenceTransformer.__init__

    def _patched(self, *a, **kw):
        kw["device"] = "cpu"
        return _orig_init(self, *a, **kw)

    _st.SentenceTransformer.__init__ = _patched


def _load() -> None:
    global _READY, _LOAD_ERR, _V10, _V11
    try:
        _load_commune_stats()
        if os.getenv("OBS_FORCE_CPU_EMBED", "1") == "1":
            _force_cpu_embeddings()
        _install_patches()

        print("[obs] initialising RAPTOR v10 …")
        v10 = _v10.RaptorSubQuestionPipeline(
            chroma_path="./chroma_portrait",
            source_collection="portrait_verbatims",
            summary_collection="raptor_summaries",
        )
        v10.init()
        _V10 = v10
        print("[obs] v10 ready")
        _load_objective_stats()

        if os.getenv("OBS_ENABLE_V11", "1") == "1":
            try:
                from rag_v11_agentic import AgenticRAGPipeline
                v11 = AgenticRAGPipeline(chroma_path="./chroma_portrait")
                v11.init()
                _V11 = v11
                if "v11" not in VERSION_CHOICES:
                    VERSION_CHOICES.append("v11")
                print("[obs] v11 ready")
            except Exception as exc:                               # noqa: BLE001
                print(f"[obs] v11 unavailable: {exc}")

        _READY = True
        print("[obs] pipeline loaded — accepting queries")
    except Exception as exc:                                       # noqa: BLE001
        _LOAD_ERR = f"{exc}\n{traceback.format_exc()}"
        print(f"[obs] LOAD FAILED:\n{_LOAD_ERR}")


def start_loading() -> None:
    threading.Thread(target=_load, name="obs-load", daemon=True).start()


def status() -> dict:
    return {"ready": _READY, "error": _LOAD_ERR, "v11": _V11 is not None}


# --------------------------------------------------------------------------- #
# Parallel sub-questions — a DELIBERATE DUPLICATE of
# RaptorSubQuestionPipeline.query() (rag_v10_raptor_subq.py:786-975), not a
# patch. rag_v10_raptor_subq.py is never edited; this exists purely because
# a caller-side for-loop can't be parallelised by patching the functions it
# calls (retrieval + answer per sub-question are sequential inside the
# original method, so making them concurrent means owning that loop).
#
# Every branch here mirrors the original line for line — same bypass check
# for "Corse entière" questions, same OppChoVec pre-injection, same bilan/
# sources_mobilisees handling — with ONLY the per-sub-question retrieve+answer
# loop turned into a thread pool. If rag_v10_raptor_subq.py's query() ever
# changes, this needs updating too; it is not derived automatically.
#
# Set OBS_PARALLEL_SUBQ=0 to fall back to the untouched, fully sequential
# pipeline.query() instead of this function (see run(), below).
# --------------------------------------------------------------------------- #
_PARALLEL_SUBQ = os.getenv("OBS_PARALLEL_SUBQ", "1") == "1"
_PARALLEL_SUBQ_WORKERS = int(os.getenv("OBS_PARALLEL_SUBQ_WORKERS", "4"))


def _parallel_query(pipeline, question: str, k: int = 5, n_subquestions: int = None,
                    extra_context: str = "", force_mixed: bool = False,
                    use_bilan: bool = True, no_typing: bool = False,
                    temperature_override=None):
    if n_subquestions is None:
        n_subquestions = _v10.DEFAULT_N_SUBQUESTIONS
    if not pipeline._initialized:
        raise RuntimeError("Pipeline non initialise. Appelez init() d'abord.")

    print(f"\n[v10-parallel] Question : {question}")

    # --- Détection question globale Corse (identique à v10 ; un seul "sous-
    # entretien" ici, rien à paralléliser sur cette branche) ---
    q_norm = "".join(c for c in unicodedata.normalize("NFD", question.lower())
                     if unicodedata.category(c) != "Mn")
    try:
        from commune_detector import detect_communes as _dc
        communes_in_q = _dc(question)
    except ImportError:
        communes_in_q = []
    is_global_q = not communes_in_q and any(kw in q_norm for kw in (
        "moyen", "moyenne", "general", "global", "ensemble", "niveau",
        "corse entiere", "ile entiere", "l ensemble", "toutes les communes",
        "score global", "score corse", "indicateur corse",
    ))
    if is_global_q:
        context_str, sources = pipeline.retriever.query(question, k=k)
        global_extra = extra_context
        if pipeline.retriever._oppchovec:
            try:
                agg = pipeline.retriever._oppchovec.get(
                    ids=["oppchovec_aggregate_corse"], include=["documents", "metadatas"])
                if agg["documents"]:
                    global_extra = ("[Scores OppChoVec — Corse entière (indicateurs territoriaux objectifs)]\n"
                                    + agg["documents"][0] + ("\n\n" + extra_context if extra_context else ""))
            except Exception:
                pass
        if global_extra:
            context_str = global_extra + "\n\n" + context_str
        single_answer = _v10.answer_subquestion(question, context_str, temperature_override=temperature_override)
        single_pair = [(question, single_answer)]
        global_raw = _v10.synthesize_answers(
            question, single_pair,
            source_bilan={1: {"has_subjective": True, "has_objective": True}},
            use_bilan=use_bilan, temperature_override=temperature_override)
        final_answer, global_sm = _v10._parse_sources_mobilisees(global_raw)
        scoring = {"applicable": False, "dimension": None, "score": None,
                   "justification": "Question globale — scoring non applicable"}
        sub_qa_pairs_out = [{"idx": 1, "question": question, "answer": single_answer}]
        return final_answer, sources, scoring, sub_qa_pairs_out, global_sm

    # --- Etape 1 : Decomposition (identique) ---
    try:
        sub_questions = _v10.decompose_question(
            question, n=n_subquestions, extra_context=extra_context, force_mixed=force_mixed,
            no_typing=no_typing, temperature_override=temperature_override)
    except RuntimeError:
        refusal = _v10._call_mistral(
            f"Question : {question}",
            "Tu es un assistant spécialisé en qualité de vie en Corse. "
            "Cette question ne relève pas de ton domaine d'expertise. "
            "Réponds poliment que tu ne peux pas répondre à cette question.",
            max_tokens=300, temperature=temperature_override if temperature_override is not None else 0.3)
        empty_scoring = {"applicable": False, "dimension": None, "score": None,
                         "justification": "Question hors-domaine"}
        return refusal, [], empty_scoring, [], []

    # --- Etape 1bis : pré-injection OppChoVec (identique) ---
    opp_extra = ""
    if pipeline.retriever._oppchovec and pipeline.retriever._is_ranking_question(question):
        try:
            cl = pipeline.retriever._oppchovec.get(
                ids=["oppchovec_classement_global"], include=["documents", "metadatas"])
            if cl["documents"]:
                opp_extra = ("[Classement OppChoVec des communes corses — référence pour filtrer par EPCI/commune]\n"
                             + cl["documents"][0][:8000])
        except Exception:
            pass
    if communes_in_q and pipeline.retriever._oppchovec:
        try:
            q_emb = pipeline.retriever._encode_query(question)
            for com in communes_in_q[:2]:
                res_c = pipeline.retriever._oppchovec.query(
                    query_embeddings=[q_emb], n_results=1,
                    where={"$and": [{"source": {"$in": ["oppchovec_betti_0_10", "oppchovec_aggregate"]}},
                                     {"commune": {"$eq": com}}]},
                    include=["documents", "metadatas", "distances"])
                if res_c["documents"][0]:
                    opp_extra += f"\n\n[Scores OppChoVec — {com}]\n{res_c['documents'][0][0][:1500]}"
            opp_extra = opp_extra.strip()
        except Exception:
            pass

    # --- Etape 2 : Retrieval + reponse par sous-question — PARALLELISE ICI ---
    # (the only real change from the original: this was a plain `for` loop)
    def _one(i, sq):
        context_str, sources = pipeline.retriever.query(sq, k=k)
        merged_extra = "\n\n".join(x for x in [opp_extra, extra_context] if x)
        if merged_extra:
            context_str = merged_extra + "\n\n" + context_str
        ans = _v10.answer_subquestion(sq, context_str, temperature_override=temperature_override)
        for s in sources:
            s["sub_question_idx"] = i + 1
            s["sub_question"] = sq
        return sq, ans, sources

    n = len(sub_questions)
    results = [None] * n
    workers = max(1, min(n, _PARALLEL_SUBQ_WORKERS)) if n else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, i, sq): i for i, sq in enumerate(sub_questions)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    sub_qa_pairs = [(sq, ans) for sq, ans, _s in results]
    all_sources = []
    for _sq, _ans, sources in results:
        all_sources.extend(sources)

    # --- Bilan déterministe des sources par sous-question (identique) ---
    source_bilan = {}
    if use_bilan:
        for s in all_sources:
            idx = s.get("sub_question_idx", 0)
            if idx not in source_bilan:
                source_bilan[idx] = {"has_subjective": False, "has_objective": False}
            t = s.get("type", "") or s.get("source_type", "")
            if "raptor" in t or "enquete" in t or "verbatim" in t:
                source_bilan[idx]["has_subjective"] = True
            if "opp" in t or "objectif" in t or "equipement" in t:
                source_bilan[idx]["has_objective"] = True

    # --- Etape 3 : Synthese finale (identique) ---
    sources_per_subq = {}
    for s in all_sources:
        idx = s.get("sub_question_idx", 0)
        sources_per_subq.setdefault(idx, []).append(s)
    final_answer_raw = _v10.synthesize_answers(
        question, sub_qa_pairs, source_bilan,
        use_bilan=use_bilan, sources_per_subq=sources_per_subq,
        temperature_override=temperature_override)
    final_answer, sources_mobilisees = _v10._parse_sources_mobilisees(final_answer_raw)

    # --- Etape 4 : Notation de la dimension (identique) ---
    scoring = _v10.score_dimension(question, final_answer)

    sub_qa_list = [{"idx": i + 1, "question": sq, "answer": ans}
                   for i, (sq, ans) in enumerate(sub_qa_pairs)]
    return final_answer, all_sources, scoring, sub_qa_list, sources_mobilisees


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
def run(question: str, cfg: dict, emit: Callable[[dict], None], history: list = None) -> dict:
    """Run one query. Blocks for the whole pipeline (~40-120 s).

    `history` (optional): prior turns from the client's own thread, used only
    to condense a follow-up into a standalone question before it ever reaches
    v10 — see _condense_question(). v10 itself stays single-turn/stateless.
    """
    global _CUR

    if not _READY:
        raise NotReady("The assistant is still loading its data. Please wait a moment.")
    if not _LOCK.acquire(blocking=False):
        raise Busy("The assistant is currently answering another question.")

    try:
        _CUR = {
            "emit": emit,
            "steps": {},
            "stage": "init",
            "answer_i": 0,
            "n_sub": cfg["n_subquestions"],
            "decomposer": cfg["decomposer"],
            "answerer": cfg["answerer"],
            "synthesizer": cfg["synthesizer"],
            "lang": cfg.get("output_language", "fr"),
        }

        # The right-hand commune selector is CONTEXT ONLY (sidebar indicators).
        # It must not steer the answer — the question goes to the pipeline
        # verbatim and v10's own commune_detector decides what it's about.
        commune = (cfg.get("commune") or "").strip()

        # Conversation memory: fold prior turns into a standalone question
        # *before* v10 sees anything. v10's query() below is unchanged and
        # unaware this ever happened — it just gets a plain question string.
        asked = _condense_question(question, history) if history else question

        t0 = time.time()
        version = cfg["version"]

        if version == "v11" and _V11 is not None:
            answer, sources, sub_qa = _V11.query(asked, k=cfg["k"], use_fast_path=True)
            scoring = {"applicable": False}
            sources_mob = None
        else:
            version = "v10"
            extra = _v10_extra_context(_V10, asked)
            v10_query = _parallel_query if _PARALLEL_SUBQ else _V10.query
            v10_args = ((_V10, asked) if _PARALLEL_SUBQ else (asked,))
            answer, sources, scoring, sub_qa, sources_mob = v10_query(
                *v10_args,
                k=cfg["k"],
                n_subquestions=cfg["n_subquestions"],
                extra_context=extra["ctx"],
                force_mixed=extra["bieneetre"],
                temperature_override=cfg["temperature"],
            )

        elapsed = time.time() - t0
        steps = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()}
                 for k, v in _CUR["steps"].items()}

        return {
            "answer": _polish_answer(answer),
            "sources": _clean_sources(sources),
            "sub_questions": sub_qa,
            "sources_mobilisees": sources_mob,
            "scoring": scoring,
            "meta": {
                "elapsed_s": round(elapsed, 1),
                "version": version,
                "commune": commune or None,
                "original_question": question,
                "standalone_question": asked if asked != question else None,
                "parallel_subq": _PARALLEL_SUBQ if version == "v10" else None,
                "decomposer": cfg["decomposer"],
                "answerer": cfg["answerer"],
                "synthesizer": cfg["synthesizer"],
                "output_language": cfg.get("output_language", "fr"),
                "k": cfg["k"],
                "n_subquestions": cfg["n_subquestions"],
                "temperature": cfg["temperature"],
                "steps": steps,
            },
        }
    finally:
        _CUR = None
        _LOCK.release()
