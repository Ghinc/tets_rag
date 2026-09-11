"""
rag_v12_local.py — Pipeline RAG v12 : backend LLM commutable API / local

Architecture :
  - rag_v10_raptor_subq.py reste intact (référence).
  - Ce module monkey-patche _call_claude(), _call_mistral() et score_dimension()
    dans v10 pour brancher le backend choisi via LLM_BACKEND=api|local.
  - Chaque appel LLM capture timing et usage tokens (prompt + completion).
  - Métriques sauvegardées en JSON par question via get_last_metrics().

LLM_BACKEND=api  (défaut) : SDK Anthropic + Mistral API — validation + baseline.
LLM_BACKEND=local          : llama-server (activer quand GPU réglé).

Points de contrôle Phase 0 :
  🛑 Valider baseline LLM_BACKEND=api avant de basculer sur local.
  🛑 Corriger GPU (memory clock) avant d'activer LLM_BACKEND=local.
"""

import os
import re
import time
import unicodedata
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_v10_raptor_subq as _v10

try:
    from local.hallucination_guard import (
        HallucinationGuard as _HGuard,
        build_silence_list_phase1 as _build_silence_p1,
        build_silence_list_baseline as _build_silence_base,
        build_silence_header as _silence_header,
        build_silence_map_from_chroma as _build_silence_map,
        build_silence_header_with_map as _silence_header_map,
        strip_block_tags as _strip_block_tags,
    )
    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False
    print("[v12] hallucination_guard non disponible — garde désactivée")

# ── Service d'embedding distant (évite le chargement local de bge-m3) ────────

EMBEDDER_SERVICE_URL = os.getenv("EMBEDDER_SERVICE_URL", "http://localhost:8765")


def _check_embedder_service() -> str | None:
    """Retourne l'URL du service si disponible, None sinon."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{EMBEDDER_SERVICE_URL}/health", timeout=2) as r:
            if r.status == 200:
                return EMBEDDER_SERVICE_URL
    except Exception:
        pass
    return None


def _inject_remote_st(svc_url: str) -> None:
    """Injecte un faux module sentence_transformers utilisant le service HTTP.

    Cela permet à RaptorRetriever.init() de faire `SentenceTransformer("BAAI/bge-m3")`
    sans charger le modèle localement.
    """
    if "sentence_transformers" in sys.modules:
        return  # déjà chargé (api_server l'a préchargé), pas besoin de remplacer

    import types
    import numpy as np

    class _RemoteST:
        """Drop-in SentenceTransformer qui délègue encode() au service HTTP."""
        def __init__(self, model_name_or_path: str, **kw):
            self._svc = svc_url
            self.max_seq_length = 8192

        def encode(self, sentences, batch_size: int = 32,
                   normalize_embeddings: bool = True, **kw):
            import urllib.request, urllib.error, json
            single = isinstance(sentences, str)
            if single:
                sentences = [sentences]
            body = json.dumps({
                "texts": sentences,
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
            }).encode()
            req = urllib.request.Request(
                f"{self._svc}/encode",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            result = np.array(data["embeddings"])
            return result[0] if single else result

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _RemoteST
    fake_st.__version__ = f"remote@{svc_url}"
    sys.modules["sentence_transformers"] = fake_st
    print(f"[v12] sentence_transformers → service distant {svc_url}")


# ── Configuration backend ────────────────────────────────────────────────────

LLM_BACKEND        = os.getenv("LLM_BACKEND", "api")
LLAMA_BASE_URL     = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")
LLAMA_MODEL        = os.getenv("LLAMA_MODEL", "Qwen3.5-9B-Q4_K_M")
PHASE1_STRUCTURED  = os.getenv("PHASE1_STRUCTURED", "0") == "1"
_LLAMA_SEED        = 42
# Thinking désactivé partout : --reasoning-budget 0 au lancement de llama-server.
# _TASKS_THINK vide = /no_think + enable_thinking=False sur TOUS les appels locaux.
_TASKS_THINK: set  = set()
# Plafond dur pour les tests à GPU lent (LOCAL_MAX_TOKENS_CAP, 0 = pas de plafond).
_LOCAL_MAX_TOKENS_CAP: int = int(os.getenv("LOCAL_MAX_TOKENS_CAP", "0"))

# Références aux originaux capturées AVANT tout patch (évite la récursion infinie).
_ORIG_CALL_CLAUDE  = _v10._call_claude
_ORIG_CALL_MISTRAL = _v10._call_mistral

_llama_client = None


def _get_llama_client():
    global _llama_client
    if _llama_client is None:
        from openai import OpenAI
        import httpx
        # Timeout long : synthesis avec thinking peut prendre 3-5 min.
        _llama_client = OpenAI(
            api_key="local",
            base_url=LLAMA_BASE_URL,
            http_client=httpx.Client(timeout=httpx.Timeout(7200.0, connect=10.0)),
        )
    return _llama_client


# ── Contexte d'instrumentation ───────────────────────────────────────────────
# _CURRENT_STEP est mis à jour par query() avant chaque appel LLM.
# Les fonctions patchées y lisent l'étape courante pour accumuler les métriques.

_CURRENT_STEP: str = "unknown"
_STEP_ACCUM: dict = {}   # {step: {elapsed_s, prompt_tokens, completion_tokens, n_calls}}

# Grammaire GBNF injectée sur l'appel synthesize en mode silence.
# Vide = pas de contrainte grammar. Setté par _synthesize_hook_p1 quand silence actif.
_SYNTHESIS_GRAMMAR: str = ""

# Grammaire GBNF : force [[VERDICT]]...[[/VERDICT]] (+ CONTEXTE_ELARGI optionnel).
# text ::= ( [^\[] | "[" [^\[] )* — permet [SQ1] mais bloque [[  → force fermeture de bloc.
_GBNF_SILENCE_BLOCKS = r"""root ::= verdict ( whitespace contexte )?
verdict ::= "[[VERDICT]]" text "[[/VERDICT]]"
contexte ::= "[[CONTEXTE_ELARGI" scope-attr "]]" text "[[/CONTEXTE_ELARGI]]"
scope-attr ::= " scope=\"" [a-zA-Z_-]+ "\""
text ::= ( [^\[] | "[" [^\[] )*
whitespace ::= [ \t\n\r]*
"""


def _reset_accum():
    global _STEP_ACCUM
    _STEP_ACCUM = {
        "decompose":      {"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0},
        "answer":         {"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0},
        "synthesize":     {"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0},
        "score_dimension":{"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0},
    }


def _accum(step: str, elapsed_s: float, prompt_tok: int, completion_tok: int):
    if step not in _STEP_ACCUM:
        _STEP_ACCUM[step] = {"elapsed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
    _STEP_ACCUM[step]["elapsed_s"]         += elapsed_s
    _STEP_ACCUM[step]["prompt_tokens"]     += prompt_tok
    _STEP_ACCUM[step]["completion_tokens"] += completion_tok
    _STEP_ACCUM[step]["n_calls"]           += 1


# ── Vérification thinking au démarrage (LLM_BACKEND=local uniquement) ────────

# Budget additionnel ajouté à max_tokens quand le thinking reste actif.
# Qwen3.5-9B : ~100-400 tok de raisonnement ; 1500 donne de la marge.
_THINKING_ACTIVE  = False
_THINK_OVERHEAD   = 0
# Overheads par type de call mistral (thinking réel varie avec la complexité du prompt)
_THINK_OVERHEAD_DECOMPOSE  = 0
_THINK_OVERHEAD_SYNTHESIZE = 0


def check_no_think() -> None:
    """Vérifie que thinking est bien désactivé avec --jinja + chat_template_kwargs.

    Signal fiable : usage.completion_tokens < 20 ET reasoning_content vide.
    Utilise un prompt avec timestamp pour éviter les cache hits.
    """
    global _THINKING_ACTIVE
    import time as _t
    client = _get_llama_client()
    ts = int(_t.time())
    resp = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": "Tu es un assistant."},
            {"role": "user",   "content": f"Réponds uniquement 'OK'. (ts={ts})"},
        ],
        max_tokens=64,
        temperature=0.0,
        extra_body={
            "seed": ts,  # seed unique pour éviter le cache
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choice    = resp.choices[0]
    text      = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    n_tok     = resp.usage.completion_tokens if resp.usage else None

    if "<think>" in text.lower() or reasoning.strip() or n_tok is None or n_tok >= 20:
        _THINKING_ACTIVE = True
        print(
            f"[check_no_think] WARN — thinking actif malgré /no_think\n"
            f"  completion_tokens={n_tok}, reasoning_words≈{len(reasoning.split())}\n"
            f"  content={text[:60]!r}\n"
            f"  => _THINKING_ACTIVE=True : overhead decompose=+{_THINK_OVERHEAD_DECOMPOSE} / answer=+{_THINK_OVERHEAD} / synth cap 5000."
        )
    else:
        _THINKING_ACTIVE = False
        print(f"[check_no_think] OK — tokens={n_tok}, content={text.strip()!r}")


# ── Détection scope + question_type ─────────────────────────────────────────

_RE_FAISABILITE = re.compile(
    r"peut[- ]on\b|est[- ]il possible de|"
    r"les donn[eé]es permettent[- ]elles|est[- ]ce possible",
    re.IGNORECASE,
)
_RE_EXISTENCE = re.compile(
    r"existe[- ]t[- ]il des donn[eé]es|"
    r"avez[- ]vous des donn[eé]es|"
    r"y a[- ]t[- ]il des (?:donn[eé]es|informations)",
    re.IGNORECASE,
)
_GLOBAL_KW = (
    "moyen", "moyenne", "general", "global", "ensemble", "niveau",
    "corse entiere", "ile entiere", "l ensemble", "toutes les communes",
    "score global", "score corse", "indicateur corse",
)


def _detect_scope_and_type(question: str, communes_detected: list) -> tuple:
    """Retourne (scope, question_type).

    scope        ∈ {globale, locale, thematique}
    question_type ∈ {factuelle, faisabilite, existence}

    scope reprend exactement les mêmes critères que _is_global_q de v10
    (lignes 818-822), sans dupliquer la détection.
    """
    q_norm = "".join(
        c for c in unicodedata.normalize("NFD", question.lower())
        if unicodedata.category(c) != "Mn"
    )

    if not communes_detected and any(kw in q_norm for kw in _GLOBAL_KW):
        scope = "globale"
    elif communes_detected:
        scope = "locale"
    else:
        scope = "thematique"

    if _RE_FAISABILITE.search(question):
        question_type = "faisabilite"
    elif _RE_EXISTENCE.search(question):
        question_type = "existence"
    else:
        question_type = "factuelle"

    return scope, question_type


# ── Patch _call_mistral (instrumented) ──────────────────────────────────────

def _call_mistral_instrumented(prompt: str, system_prompt: str,
                               model: str = _v10.DECOMPOSER_MODEL,
                               max_tokens: int = 1000,
                               temperature: float = 0.3,
                               max_retries: int = 5) -> str:
    """Remplace _v10._call_mistral — même interface, capture timing + usage.

    LLM_BACKEND=local : route vers llama-server (Qwen3.5-9B), thinking OFF.
    LLM_BACKEND=api   : Mistral AI (comportement original).
    """
    t0 = time.time()

    if LLM_BACKEND == "local":
        client = _get_llama_client()
        sys_content  = system_prompt
        user_content = prompt
        # chat_template_kwargs désactive le thinking via le template Jinja du GGUF
        # (requis avec --jinja ; enable_thinking seul est ignoré par ce build)
        _extra = {
            "seed": _LLAMA_SEED,
            "enable_thinking": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if _CURRENT_STEP == "synthesize" and _SYNTHESIS_GRAMMAR:
            _extra["grammar"] = _SYNTHESIS_GRAMMAR
        if _THINKING_ACTIVE:
            if max_tokens >= 2000:   # synthesis
                eff_max = min(max_tokens + _THINK_OVERHEAD_SYNTHESIZE, 5000)
            else:                    # decompose
                eff_max = max_tokens + _THINK_OVERHEAD_DECOMPOSE
        else:
            # thinking désactivé : eff_max = max_tokens, cap synthesis à 2500
            eff_max = min(max_tokens, 2500) if max_tokens >= 2000 else max_tokens
        if _LOCAL_MAX_TOKENS_CAP > 0:
            eff_max = min(eff_max, _LOCAL_MAX_TOKENS_CAP)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=LLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens=eff_max,
                    temperature=temperature,
                    extra_body=_extra,
                )
                elapsed = time.time() - t0
                p_tok = c_tok = 0
                if resp.usage:
                    p_tok = resp.usage.prompt_tokens
                    c_tok = resp.usage.completion_tokens
                    _accum(_CURRENT_STEP, elapsed, p_tok, c_tok)
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
                r_tok = len(reasoning.split()) * 13 // 10 if reasoning else 0
                content = resp.choices[0].message.content or ""
                print(f"  [call] {_CURRENT_STEP}: p={p_tok} c={c_tok} "
                      f"r≈{r_tok} out={len(content)}ch eff_max={eff_max} {elapsed:.0f}s")
                return content
            except Exception as e:
                err = str(e)
                if ("429" in err or "timeout" in err.lower()
                        or "connection" in err.lower()) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"    [RETRY {attempt+1}/{max_retries}] llama-server : {err[:80]}...")
                    time.sleep(wait)
                else:
                    print(f"    [ERREUR llama-server] {err[:200]}")
                    raise
    else:
        from openai import OpenAI
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY non definie")
        client = OpenAI(api_key=api_key, base_url=_v10.MISTRAL_BASE_URL)
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - t0
                if response.usage:
                    _accum(_CURRENT_STEP, elapsed,
                           response.usage.prompt_tokens,
                           response.usage.completion_tokens)
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 2
                    print(f"    [RATE LIMIT] Mistral : attente {wait}s (tentative {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise


# ── Patch _call_claude (instrumented + backend commutable) ───────────────────

def _call_claude(prompt: str, system_prompt: str,
                 model: str = _v10.ANSWERER_MODEL,
                 max_tokens: int = 800,
                 temperature: float = 0.2,
                 max_retries: int = 5,
                 task: str = "answer") -> str:
    t0 = time.time()

    if LLM_BACKEND == "local":
        client = _get_llama_client()
        use_think  = task in _TASKS_THINK
        sys_content  = system_prompt
        user_content = prompt
        _extra: dict = {"seed": _LLAMA_SEED, "cache_prompt": True}
        if not use_think:
            _extra["enable_thinking"] = False
            _extra["chat_template_kwargs"] = {"enable_thinking": False}
        overhead = _THINK_OVERHEAD if (_THINKING_ACTIVE and not use_think) else 0
        eff_max = max_tokens + overhead
        if _LOCAL_MAX_TOKENS_CAP > 0:
            eff_max = min(eff_max, _LOCAL_MAX_TOKENS_CAP)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=LLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens=eff_max,
                    temperature=temperature,
                    extra_body=_extra,
                )
                elapsed = time.time() - t0
                p_tok = c_tok = 0
                if resp.usage:
                    p_tok = resp.usage.prompt_tokens
                    c_tok = resp.usage.completion_tokens
                    _accum(_CURRENT_STEP, elapsed, p_tok, c_tok)
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
                r_tok = len(reasoning.split()) * 13 // 10 if reasoning else 0
                content = resp.choices[0].message.content or ""
                print(f"  [call] {_CURRENT_STEP}: p={p_tok} c={c_tok} "
                      f"r≈{r_tok} out={len(content)}ch eff_max={eff_max} {elapsed:.0f}s")
                return content
            except Exception as e:
                err = str(e)
                if ("429" in err or "timeout" in err.lower()
                        or "connection" in err.lower()) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"    [RETRY {attempt+1}/{max_retries}] llama-server : {err[:80]}...")
                    time.sleep(wait)
                else:
                    print(f"    [ERREUR llama-server] {err[:200]}")
                    raise
    else:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY non definie")
        client = anthropic.Anthropic(api_key=api_key, timeout=300.0)
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                elapsed = time.time() - t0
                if response.usage:
                    _accum(_CURRENT_STEP, elapsed,
                           response.usage.input_tokens,
                           response.usage.output_tokens)
                return response.content[0].text
            except Exception as e:
                err = str(e)
                is_retryable = ("429" in err or "529" in err or "overloaded" in err.lower()
                                or "rate" in err.lower() or "timeout" in err.lower()
                                or "connection" in err.lower())
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"    [RETRY {attempt+1}/{max_retries}] Claude : {err[:80]}...")
                    time.sleep(wait)
                else:
                    print(f"    [ERREUR Claude] {err[:200]}")
                    raise


# ── score_dimension désactivée ───────────────────────────────────────────────

def _score_dimension_noop(question: str, answer: str) -> dict:
    _accum("score_dimension", 0.0, 0, 0)
    return {"applicable": False, "dimension": None, "score": None,
            "justification": "Désactivée en v12 (chemin d'inférence)"}


# ── Phase 1 : extraction structurée (answer → JSON compact) ─────────────────

_SYSTEM_ANSWERER_JSON = (
    "Tu es un analyste de données sur la qualité de vie en Corse. "
    "Ta tâche : extraire l'information pertinente du contexte et la restituer en JSON valide. "
    "UNIQUEMENT du JSON — aucun texte avant ou après, aucune balise markdown.\n\n"
    'FORMAT : {"valeur":"<donnée chiffrée/factuelle principale ou null>","contexte":'
    '"<1-2 phrases avec attribution de source et portée géographique>","source_type":'
    '"<enquete_citoyenne|oppchovec|equipements|entretiens|mixte|aucune_donnee>","confiance":'
    '"<high|medium|low>"}\n\n'
    "RÈGLES :\n"
    "• Aucune donnée → valeur:null, source_type:'aucune_donnee', "
    "contexte:'Aucune donnée disponible pour cette sous-question.', confiance:'low'\n"
    "• Attribution dans 'contexte' : précise toujours la source "
    "(enquête citoyenne / OppChoVec / équipements communaux / entretiens)\n"
    "• Portée géo : si les données couvrent la Corse entière et non la commune interrogée, "
    "l'indiquer dans 'contexte'\n"
    "• OppChoVec (0-10, 360 communes corses) : "
    "Opp=éducation/mobilité/TIC ; Cho=participation civique ; Vec=revenu/logement/emploi/services\n"
    "• N'invente aucun chiffre, score ou fait absent du contexte\n"
    "• Réponds uniquement sur la/les commune(s) mentionnée(s) dans la sous-question"
)


def _answer_structured(sub_question: str, context: str,
                       temperature_override=None, no_typing: bool = False) -> str:
    """Phase 1 : réponse JSON compacte (~100-150 tok vs ~440 tok en prose)."""
    import json as _json
    prompt = f"Contexte :\n{context[:15000]}\n\nSous-question : {sub_question}"
    # 400 tok = marge pour ~200 tok thinking (mode local) + ~200 tok JSON cible
    raw = _call_claude(prompt, _SYSTEM_ANSWERER_JSON,
                       max_tokens=400,
                       temperature=temperature_override if temperature_override is not None else 0.1)
    s = raw.strip()
    try:
        return _json.dumps(_json.loads(s), ensure_ascii=False)
    except Exception:
        m = re.search(r'\{.*?\}', s, re.DOTALL)
        if m:
            try:
                return _json.dumps(_json.loads(m.group(0)), ensure_ascii=False)
            except Exception:
                pass
        return _json.dumps({"valeur": None, "contexte": s[:300],
                            "source_type": "aucune_donnee", "confiance": "low"},
                           ensure_ascii=False)


def _sq_json_to_prose(json_str: str) -> str:
    """Convertit un JSON de sous-réponse en prose étiquetée lisible par le synthesizer."""
    import json as _json
    try:
        obj = _json.loads(json_str)
        val  = obj.get("valeur") or "N/D"
        ctx  = obj.get("contexte", "")
        src  = obj.get("source_type", "?")
        conf = obj.get("confiance", "?")
        return f"[{src}, confiance:{conf}] Valeur : {val}. {ctx}"
    except Exception:
        return json_str[:300]


# ── Monkey-patch ─────────────────────────────────────────────────────────────

def _apply_patches() -> None:
    _v10._call_mistral   = _call_mistral_instrumented
    _v10._call_claude    = _call_claude
    _v10.score_dimension = _score_dimension_noop
    mode = f"LLM_BACKEND={LLM_BACKEND!r}"
    if PHASE1_STRUCTURED:
        mode += " PHASE1_STRUCTURED=1 (answer→JSON, max_tokens=400 incl. thinking)"
    print(f"[v12] Patches appliqués — {mode}")


# ── Pipeline v12 ─────────────────────────────────────────────────────────────

class RaptorSubQuestionPipelineV12(_v10.RaptorSubQuestionPipeline):
    """Sous-classe v12 : backend commutable + instrumentation par étape.

    Signature de retour étendue :
      (final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics)

    Ne pas brancher sur api_server_multi_version.py avant Phase 1.
    """

    def init(self) -> "RaptorSubQuestionPipelineV12":
        _apply_patches()

        # ── Mode service distant (bge-m3 chargé une fois dans embedder_service.py) ──
        # Vérifier en premier, AVANT tout import sentence_transformers, pour éviter
        # les conflits DLL sur Windows (Cortex XDR, chargement cold-start de 30 s+).
        _svc = _check_embedder_service()
        if _svc:
            _inject_remote_st(_svc)   # injecte module factice dans sys.modules
            if LLM_BACKEND == "local":
                check_no_think()
            super().init()            # SentenceTransformer("bge-m3") → _RemoteST
            print(f"[v12] bge-m3 → service {_svc}")
            return self

        # ── Mode local (fallback) ─────────────────────────────────────────────────
        # Import AVANT check_no_think() pour éviter conflit DLL pyarrow/openai.
        import sentence_transformers as _st_mod
        if LLM_BACKEND == "local":
            check_no_think()
        _orig_st_init = _st_mod.SentenceTransformer.__init__
        def _force_cpu_init(self_st, *a, **kw):
            kw['device'] = 'cpu'
            return _orig_st_init(self_st, *a, **kw)
        _st_mod.SentenceTransformer.__init__ = _force_cpu_init
        try:
            super().init()
        finally:
            _st_mod.SentenceTransformer.__init__ = _orig_st_init
        print("[v12] bge-m3 → CPU local")
        return self

    def query(self, question: str, **kwargs):
        global _CURRENT_STEP
        _reset_accum()

        try:
            from commune_detector import detect_communes as _dc
            communes = _dc(question)
        except ImportError:
            communes = []

        scope, question_type = _detect_scope_and_type(question, communes)
        print(f"[v12] scope={scope!r}  question_type={question_type!r}")

        t_total = time.time()

        # Le pipeline v10 appelle les étapes dans l'ordre :
        # 1. decompose_question  → _call_mistral (max_tokens=600)
        # 2. answer_subquestion × N → _call_claude (max_tokens=800)
        # 3. synthesize_answers  → _call_mistral (max_tokens=6000)
        # 4. score_dimension     → noop
        # On pilote _CURRENT_STEP avant chaque étape via un hook.

        # Patch temporaire sur decompose_question pour injecter le step label.
        _orig_decompose  = _v10.decompose_question
        _orig_answer     = _v10.answer_subquestion
        _orig_synthesize = _v10.synthesize_answers

        def _decompose_hook(*a, **kw):
            global _CURRENT_STEP
            _CURRENT_STEP = "decompose"
            return _orig_decompose(*a, **kw)

        def _answer_hook(*a, **kw):
            global _CURRENT_STEP
            _CURRENT_STEP = "answer"
            return _orig_answer(*a, **kw)

        def _answer_hook_p1(sub_question, context,
                            temperature_override=None, no_typing=False):
            global _CURRENT_STEP
            _CURRENT_STEP = "answer"
            return _answer_structured(sub_question, context, temperature_override, no_typing)

        # État partagé entre les hooks et query() — dict mutable (pas besoin de nonlocal)
        _hook_state: dict = {}

        def _synthesize_hook(*a, **kw):
            global _CURRENT_STEP
            _CURRENT_STEP = "synthesize"
            if len(a) >= 2 and isinstance(a[1], (list, tuple)):
                ctx = "\n".join(f"Q: {sq}\nA: {str(ans)}" for sq, ans in a[1])
                _hook_state["synth_context"] = ctx
                ctx_chars  = len(ctx)
                ctx_tokens = ctx_chars // 4  # estimation rapide
                print(f"[GUARD][synth] baseline — ctx={ctx_chars} chars (~{ctx_tokens} tokens), n_sq={len(a[1])}")
            else:
                print(f"[GUARD][synth] baseline — CONTEXTE NON CAPTURÉ (a={len(a)}, type a[1]={type(a[1]) if len(a)>=2 else 'N/A'})")
            return _orig_synthesize(*a, **kw)

        def _synthesize_hook_p1(initial_question, sub_qa_pairs,
                               source_bilan=None, **kw):
            global _CURRENT_STEP, _SYNTHESIS_GRAMMAR
            _CURRENT_STEP = "synthesize"
            _SYNTHESIS_GRAMMAR = ""  # reset avant de décider
            prose_pairs = [(sq, _sq_json_to_prose(ans)) for sq, ans in sub_qa_pairs]
            ctx = "\n".join(f"Q: {sq}\nA: {prose}" for sq, prose in prose_pairs)
            ctx_chars  = len(ctx)
            ctx_tokens = ctx_chars // 4
            print(f"[GUARD][synth] phase1  — ctx={ctx_chars} chars (~{ctx_tokens} tokens), n_sq={len(sub_qa_pairs)}")
            _hook_state["synth_context"] = ctx
            # Règle 1 : construire la liste de silence et l'injecter dans la question
            augmented_q = initial_question
            if _GUARD_AVAILABLE and communes:
                silence = _build_silence_p1(sub_qa_pairs, communes)
                _hook_state["silence_list"] = silence
                # Option B : vérification par registre dans ChromaDB
                try:
                    silence_map = _build_silence_map(communes)
                    _hook_state["silence_map"] = silence_map
                except Exception as _sme:
                    silence_map = {}
                    print(f"[GUARD] build_silence_map erreur : {_sme}")
                if silence_map:
                    header = _silence_header_map(silence_map)
                    if header:
                        augmented_q = header + initial_question
                        _SYNTHESIS_GRAMMAR = _GBNF_SILENCE_BLOCKS
                        print(f"[GUARD] Injection silence_map (par registre) : {silence_map}")
                        print(f"[GUARD] GBNF grammar activée pour la synthèse")
                elif silence:
                    header = _silence_header(silence)
                    augmented_q = header + initial_question
                    _SYNTHESIS_GRAMMAR = _GBNF_SILENCE_BLOCKS
                    print(f"[GUARD] Injection silence (LLM-based) pour : {silence}")
                    print(f"[GUARD] GBNF grammar activée pour la synthèse")
            return _orig_synthesize(augmented_q, prose_pairs,
                                    source_bilan, **kw)

        _v10.decompose_question = _decompose_hook
        _v10.answer_subquestion = _answer_hook_p1   if PHASE1_STRUCTURED else _answer_hook
        _v10.synthesize_answers = _synthesize_hook_p1 if PHASE1_STRUCTURED else _synthesize_hook
        _CURRENT_STEP = "score_dimension"   # fallback pour score_dimension noop

        try:
            final_answer, sources, scoring, sub_qa, sources_mob = super().query(
                question, **kwargs
            )
        finally:
            # Restaurer les originaux patchés (hooks step-label seulement)
            _v10.decompose_question = _orig_decompose
            _v10.answer_subquestion = _orig_answer
            _v10.synthesize_answers = _orig_synthesize

        total_elapsed = time.time() - t_total

        # ── Hallucination Guard ──────────────────────────────────────────────
        guard_dict: dict = {"available": False}
        if _GUARD_AVAILABLE:
            try:
                guard = _HGuard()
                synth_ctx = _hook_state.get("synth_context", "")

                if PHASE1_STRUCTURED:
                    # La liste de silence a été construite dans le hook (communes détectés)
                    silence_list = _hook_state.get("silence_list", [])
                    silence_map  = _hook_state.get("silence_map", {})
                else:
                    # Baseline : déduire depuis les sources récupérées
                    silence_list = _build_silence_base(communes, sources) if communes else []
                    silence_map  = {}

                report = guard.run_all(final_answer, silence_list, synth_ctx, sources,
                                       silence_map=silence_map)
                guard.print_report(report, prefix="")
                guard_dict = report.to_dict()
                guard_dict["available"] = True
            except Exception as _ge:
                print(f"[GUARD] Erreur lors de l'exécution : {_ge}")
                guard_dict = {"available": True, "error": str(_ge)}

        metrics = {
            "question":     question,
            "scope":        scope,
            "question_type": question_type,
            "steps":        {k: dict(v) for k, v in _STEP_ACCUM.items()},
            "total_elapsed_s":       round(total_elapsed, 2),
            "total_prompt_tokens":   sum(v["prompt_tokens"]     for v in _STEP_ACCUM.values()),
            "total_completion_tokens": sum(v["completion_tokens"] for v in _STEP_ACCUM.values()),
            "guard": guard_dict,
        }

        return _strip_block_tags(final_answer), sources, scoring, sub_qa, sources_mob, scope, question_type, metrics


def get_last_metrics() -> dict:
    """Retourne une copie des métriques de la dernière question traitée."""
    return {k: dict(v) for k, v in _STEP_ACCUM.items()}


# ── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="RAG v12 : backend commutable API/local")
    parser.add_argument("--query",   required=True)
    parser.add_argument("--k",       type=int, default=5)
    parser.add_argument("--backend", default=None, help="Surcharge LLM_BACKEND")
    args = parser.parse_args()

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    pipeline = RaptorSubQuestionPipelineV12()
    pipeline.init()

    final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics = \
        pipeline.query(args.query, k=args.k)

    print("\n" + "=" * 70)
    print(f"scope={scope}  question_type={question_type}")
    print("=" * 70)
    print(final_answer)
    print("\n--- MÉTRIQUES ---")
    print(_json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n=== SOURCES ({len(sources)}) ===")
    for s in sources[:5]:
        print(f"  [{s.get('sub_question_idx','?')}] {s.get('type', s.get('source_type','?'))} — "
              f"{s.get('extrait','')[:80]}...")
