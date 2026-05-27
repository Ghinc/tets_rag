"""
RAG v11 - Agentic RAG : boucle ReAct + CRAG gate + fast path v9.

Architecture :
  - Classificateur de complexite (regex rapide)
  - Fast path : questions simples → v9 direct + Mistral Small
  - Boucle ReAct (Anthropic Tool Use natif, Claude Sonnet) pour questions complexes
  - 5 outils : summary_search, verbatim_search, score_lookup, geo_neighbors, decompose
  - CRAG gate : cosine similarity BGE-M3 pour evaluer la pertinence du contexte recupere

Usage:
    python rag_v11_agentic.py --query "Quelles dimensions ressortent chez les agriculteurs ?"
    python rag_v11_agentic.py --query "Quel est le score OppChoVec d'Ajaccio ?" --no-fast-path
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Constantes
# ============================================================

EXECUTOR_MODEL = "claude-sonnet-4-6"
FAST_MODEL     = "mistral-small-latest"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
CHROMA_PATH    = "./chroma_portrait"
MAX_ITERATIONS = 5

# Mots-cles hors domaine
_OFF_TOPIC_KW = [
    "pib", "météo", "recette", "cuisine", "sport", "football",
    "histoire de france", "election nationale", "president", "impot",
    "bourse", "action", "crypto", "bitcoin",
]

# Patterns de questions simples (factuel direct)
_SIMPLE_PATTERNS = [
    re.compile(r"(quel est le score|quelle commune|score .{1,30}(ajaccio|bastia|corse|sartène|calvi|corte))", re.I),
    re.compile(r"(meilleure commune|commune .{1,20}meilleur|meilleur score|plus (haut|elevé|faible|bas))", re.I),
    re.compile(r"(ajaccio .{1,20}(supérieur|inférieur)|bastia .{1,20}(supérieur|inférieur))", re.I),
    re.compile(r"(quel rang|classement|quelle place|quelle est la commune)", re.I),
    re.compile(r"(oppchovec|opp ?chovec|score .{1,20}bien-être)\s*(d'|de |a |à )\w", re.I),
]

# Mots-cles OppChoVec (identique api_server)
_OPPCHOVEC_KEYWORDS = [
    "oppchovec", "score", "scores",
    "opportunités", "opportunite", "opportunites",
    "choix", "vécu", "vecu",
    "indice", "indicateur", "indicateurs",
    "classement", "classé", "classee", "rang",
    "bien placé", "bien place", "mal placé", "mal place",
    "meilleur", "moins bon", "moins bonne",
    "mieux noté", "moins bien noté",
]

# Definition des 5 outils Anthropic
TOOLS = [
    {
        "name": "summary_search",
        "description": (
            "Recherche dans les résumés RAPTOR hiérarchiques : synthèses analytiques pré-calculées "
            "par groupe démographique (âge, profession, commune, dimension QdV). "
            "Idéal pour les questions qualitatives générales, perceptions, tendances. "
            "Utilise cet outil en premier pour les questions ouvertes. "
            "Sources : verbatims et entretiens citoyens (bien-être subjectif)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Requête de recherche en français"},
                "k": {"type": "integer", "description": "Nombre de résultats (défaut 7)", "default": 7},
            },
            "required": ["query"],
        },
    },
    {
        "name": "verbatim_search",
        "description": (
            "Recherche dans les verbatims citoyens bruts avec filtres démographiques optionnels. "
            "Idéal pour des questions ciblées sur un groupe spécifique (âge, profession, commune, dimension). "
            "Utilise les filtres pour affiner si la question mentionne un groupe précis. "
            "Sources : verbatims bruts citoyens (bien-être subjectif)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Requête de recherche"},
                "filters": {
                    "type": "object",
                    "description": "Filtres optionnels pour restreindre la recherche",
                    "properties": {
                        "commune": {"type": "string", "description": "Nom exact de la commune"},
                        "age_range": {
                            "type": "string",
                            "description": "Tranche d'âge",
                            "enum": ["15-24", "25-34", "35-49", "50-64", "65+"],
                        },
                        "profession": {"type": "string", "description": "Catégorie socio-professionnelle"},
                        "dimension": {"type": "string", "description": "Dimension de qualité de vie"},
                    },
                },
                "k": {"type": "integer", "description": "Nombre de verbatims (défaut 8)", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "score_lookup",
        "description": (
            "Recherche les scores OppChoVec (Opportunités / Choix / Vécu) par commune. "
            "Utiliser pour toute question sur des indicateurs chiffrés de bien-être, "
            "classements, comparaisons de scores entre communes. "
            "Sources : indicateurs OppChoVec (bien-être OBJECTIF — données territoriales indépendantes des perceptions citoyennes)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Commune ou indicateur à rechercher (ex: 'Ajaccio OppChoVec')"},
                "k": {"type": "integer", "description": "Nombre de résultats (défaut 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "geo_neighbors",
        "description": (
            "Recherche des informations géographiques sur des communes corses "
            "(localisation, EPCI, type rural/urbain, intercommunalité). "
            "Utile pour les questions géographiques ou pour identifier des communes rurales/urbaines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commune": {
                    "type": "string",
                    "description": "Nom de la commune ou requête géographique (ex: 'communes rurales', 'EPCI Ajaccio')",
                },
                "k": {"type": "integer", "description": "Nombre de résultats (défaut 8)", "default": 8},
            },
            "required": ["commune"],
        },
    },
    {
        "name": "enquete_scores_search",
        "description": (
            "Recherche les scores moyens par dimension issus de l'enquête qualité de vie, agrégés par commune. "
            "Ces scores (1-5) couvrent : Transports, Éducation, Réseaux téléphoniques, Institutions, "
            "Tourisme, Sécurité, Santé, Situation pro, Revenus, Temps travail/perso, Logement, "
            "Services locaux, Culture, Soutien social, Vie associative, Environnement. "
            "Utiliser pour toute question sur la satisfaction des habitants vis-à-vis d'une dimension "
            "précise (ex: 'comment les habitants évaluent les transports ?', 'score santé Bastia'). "
            "Sources : scores de satisfaction déclarée par dimension (bien-être SUBJECTIF perçu — distinct des indicateurs OppChoVec objectifs)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Dimension ou commune à rechercher (ex: 'transports Ajaccio', 'environnement', 'santé')",
                },
                "k": {"type": "integer", "description": "Nombre de communes (défaut 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "decompose",
        "description": (
            "Décompose une question complexe en sous-questions indépendantes et complémentaires. "
            "Utilise cet outil quand la question est multi-aspects et qu'une recherche directe "
            "ne suffit pas à couvrir tous les angles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "La question à décomposer"},
                "n": {
                    "type": "integer",
                    "description": "Nombre de sous-questions à générer (2-5, défaut 3)",
                    "default": 3,
                },
            },
            "required": ["question"],
        },
    },
]

_SYSTEM_EXECUTOR = (
    "Tu es un assistant spécialisé dans l'analyse de la qualité de vie en Corse, "
    "s'appuyant sur des verbatims citoyens et des indicateurs territoriaux (scores OppChoVec). "
    "Tu as accès à plusieurs outils de recherche. Utilise-les de façon ciblée et itérative "
    "pour construire un contexte suffisant avant de répondre. "
    "\n\nCONTEXTE DES DONNÉES :\n"
    "- Les synthèses RAPTOR et verbatims (subjectif/quali) sont issus de l'enquête citoyenne menée auprès des habitants de Corse. "
    "Si on te demande ce que disent les habitants ou les résultats de l'enquête citoyenne, utilise summary_search ou verbatim_search.\n"
    "- enquete_scores_search → scores de satisfaction déclarés dans cette même enquête citoyenne (bien-être SUBJECTIF)\n"
    "- score_lookup → scores OppChoVec (bien-être OBJECTIF, indépendant des opinions) — "
    "C'EST L'INDICATEUR OBJECTIF DE RÉFÉRENCE ; pour toute question sur les indicateurs objectifs "
    "ou une comparaison quanti/quali, utilise cet outil et commente les résultats. "
    "Ne jamais conclure à l'absence d'indicateurs objectifs sans avoir utilisé score_lookup.\n"
    "- Les données d'équipements communaux (médecins, écoles, commerces, taux d'activité…) "
    "peuvent aussi apparaître dans le contexte via d'autres outils — elles constituent des indicateurs objectifs complémentaires.\n"
    "\nGuide d'utilisation :\n"
    "- Pour les questions qualitatives générales : commence par summary_search\n"
    "- Pour les questions sur un groupe précis : utilise verbatim_search avec filtres\n"
    "- Pour les scores OppChoVec (bien-être territorial) : utilise score_lookup\n"
    "- Pour les scores de satisfaction par dimension (transports, santé, logement…) : utilise enquete_scores_search\n"
    "- Pour les questions géographiques : utilise geo_neighbors\n"
    "- Pour les questions complexes multi-aspects : utilise decompose puis traite les sous-questions\n"
    "- Arrête de chercher quand tu as suffisamment d'information pour répondre avec confiance\n"
    "\nDISTINCTION IMPORTANTE :\n"
    "- score_lookup → bien-être OBJECTIF (indicateurs territoriaux OppChoVec, indépendants des opinions)\n"
    "- enquete_scores_search → bien-être SUBJECTIF (satisfaction perçue et déclarée par les habitants dans l'enquête citoyenne)\n"
    "Pour une question sur le bien-être global ou les indicateurs objectifs d'une commune, "
    "utilise OBLIGATOIREMENT score_lookup + enquete_scores_search. "
    "Ne jamais répondre 'aucun indicateur objectif disponible' sans avoir utilisé score_lookup.\n"
    "ATTENTION — définitions opérationnelles des sous-indicateurs OppChoVec (ne pas surinterprèter) : "
    "Opp = éducation moyenne + diversité CSP + accessibilité mobilité + couverture TIC/haut débit. "
    "Cho = % population avec droit de vote + absence de quartiers prioritaires (QPV) — "
    "PAS une mesure de libertés individuelles au sens large. "
    "Vec = revenu fiscal moyen + qualité du logement + stabilité de l'emploi + accès aux services en <20 min. "
    "Ces scores sont des proxies (0-10, relatif aux 360 communes corses uniquement) — ne pas extrapoler.\n"
    "\nRÈGLES DE RÉDACTION — à respecter absolument dans ta réponse finale : "
    "N'utilise JAMAIS les termes techniques internes : ne mentionne pas 'RAPTOR', "
    "'subjectif/quali', 'objectif/quanti', 'OppChoVec intégré', 'synthèse RAPTOR', etc. "
    "Pour citer tes sources, utilise des formulations naturelles : "
    "'selon l'enquête citoyenne', 'les habitants interrogés estiment que', "
    "'les indicateurs territoriaux montrent que', 'selon les données géographiques'. "
    "Réponds TOUJOURS en français. "
    "Ne fabrique JAMAIS d'information absente du contexte — aucun chiffre, nom, rang ou fait inventé. "
    "Si après tes recherches le contexte reste insuffisant pour répondre à un point précis, "
    "dis-le explicitement ('je ne dispose pas de données suffisantes sur ce point'). "
    "Une réponse partielle honnête est toujours préférable à une réponse complète inventée."
)


# ============================================================
# AgenticRAGPipeline
# ============================================================

class AgenticRAGPipeline:
    """
    Pipeline RAG agentique v11.
    Boucle ReAct via Anthropic Tool Use + CRAG gate + fast path v9.
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma_path = chroma_path
        self._raptor = None        # RaptorRetriever (v9)
        self._embed_model = None   # BGE-M3 (partagé depuis raptor)
        self._chroma_client = None # chromadb.PersistentClient
        self._anthropic = None     # anthropic.Anthropic
        self._mistral_key = None   # str

    def init(self):
        """Charge les modèles et connecte aux collections ChromaDB."""
        import chromadb
        from rag_v9_raptor import RaptorRetriever

        print("  [v11] Chargement RaptorRetriever...")
        self._raptor = RaptorRetriever(
            chroma_path=self.chroma_path,
            source_collection="portrait_verbatims",
            summary_collection="raptor_summaries",
        )
        self._raptor.init()
        self._embed_model = self._raptor._embed_model  # BGE-M3 partage, pas de double chargement

        # Client Chroma pour acces direct aux collections geo
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)

        # Clients LLM
        import anthropic as _anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY (ou CLAUDE_API_KEY) non definie")
        self._anthropic = _anthropic.Anthropic(api_key=api_key)

        self._mistral_key = os.getenv("MISTRAL_API_KEY")
        if not self._mistral_key:
            raise RuntimeError("MISTRAL_API_KEY non definie")

        print("  [v11] AgenticRAGPipeline pret.")

    # --------------------------------------------------------
    # CRAG gate
    # --------------------------------------------------------

    def _crag_score(self, question: str, docs: List[str]) -> float:
        """
        Score de pertinence via cosine similarity BGE-M3.
        Retourne float dans [0, 1].
        """
        import numpy as np
        if not docs:
            return 0.0
        q_emb = self._embed_model.encode(f"query: {question}")
        doc_embs = self._embed_model.encode([f"passage: {d[:500]}" for d in docs])
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        d_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(d_norms, q_norm)
        return float(np.clip(np.mean(sims), 0.0, 1.0))

    # --------------------------------------------------------
    # Classificateur de complexite
    # --------------------------------------------------------

    def _classify(self, question: str) -> dict:
        """
        Classe la question : hors_domaine | simple | complex.
        Les questions simples (factuelles directes) sont routees vers le fast path v9.
        """
        q_lower = question.lower()

        # Hors domaine
        if any(kw in q_lower for kw in _OFF_TOPIC_KW):
            return {"complexity": "hors_domaine"}

        # Simples : question factuelle directe detectable par regex
        if any(p.search(question) for p in _SIMPLE_PATTERNS):
            return {"complexity": "simple"}

        return {"complexity": "complex"}

    # --------------------------------------------------------
    # Construction clause WHERE ChromaDB
    # --------------------------------------------------------

    def _build_verbatim_where(self, filters: dict) -> Optional[dict]:
        """Construit la clause WHERE ChromaDB pour verbatim_search."""
        conditions = []
        if filters.get("commune"):
            conditions.append({"nom": {"$eq": filters["commune"]}})
        if filters.get("age_range"):
            conditions.append({"age_range": {"$eq": filters["age_range"]}})
        if filters.get("profession"):
            conditions.append({"profession": {"$eq": filters["profession"]}})
        if filters.get("dimension"):
            conditions.append({"dimension": {"$eq": filters["dimension"]}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # --------------------------------------------------------
    # Execution des outils
    # --------------------------------------------------------

    def _execute_tool(self, name: str, params: dict) -> str:
        """Exécute un outil et retourne le résultat sous forme de string."""

        if name == "summary_search":
            try:
                context_str, _ = self._raptor.query(
                    params["query"], k=params.get("k", 7)
                )
                return context_str or "Aucun résultat trouvé dans les synthèses RAPTOR."
            except Exception as e:
                return f"Erreur summary_search : {e}"

        elif name == "verbatim_search":
            try:
                query = params["query"]
                filters = params.get("filters") or {}
                k = params.get("k", 8)
                q_emb = self._embed_model.encode(f"query: {query}").tolist()
                where = self._build_verbatim_where(filters)
                col = self._raptor._source
                n = min(k, col.count())
                if n == 0:
                    return "Collection portrait_verbatims vide."
                kwargs: dict = {
                    "query_embeddings": [q_emb],
                    "n_results": n,
                    "include": ["documents", "metadatas"],
                }
                if where:
                    kwargs["where"] = where
                results = col.query(**kwargs)
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                if not docs:
                    return "Aucun verbatim trouvé avec ces filtres."
                lines = []
                for doc, meta in zip(docs, metas):
                    label = (
                        f"[{meta.get('age_range','?')}, {meta.get('profession','?')}, "
                        f"{meta.get('nom','?')}, {meta.get('dimension','?')}]"
                    )
                    lines.append(f"{label}\n{doc[:400]}")
                return "\n\n".join(lines)
            except Exception as e:
                return f"Erreur verbatim_search : {e}"

        elif name == "score_lookup":
            try:
                query = params.get("query", "")
                k = params.get("k", 5)
                docs = self._raptor.query_oppchovec(query, k=k)
                if not docs:
                    return "Scores OppChoVec non disponibles pour cette requête."
                return "\n\n".join(d["text"] for d in docs)
            except Exception as e:
                return f"Erreur score_lookup : {e}"

        elif name == "geo_neighbors":
            try:
                commune = params.get("commune", "")
                k = params.get("k", 8)
                q_emb = self._embed_model.encode(f"query: {commune}").tolist()
                # Essayer communes_geo puis zones_epci
                for col_name in ["communes_geo", "zones_epci"]:
                    try:
                        col = self._chroma_client.get_collection(col_name)
                        n = min(k, col.count())
                        if n == 0:
                            continue
                        res = col.query(
                            query_embeddings=[q_emb],
                            n_results=n,
                            include=["documents", "metadatas"],
                        )
                        docs = res["documents"][0]
                        if docs:
                            return f"[{col_name}]\n" + "\n\n".join(docs)
                    except Exception:
                        continue
                return "Aucune information géographique trouvée."
            except Exception as e:
                return f"Erreur geo_neighbors : {e}"

        elif name == "enquete_scores_search":
            try:
                query = params.get("query", "")
                k = params.get("k", 5)
                q_emb = self._embed_model.encode(f"query: {query}").tolist()
                col = self._chroma_client.get_collection("enquete_scores_commune")
                n = min(k, col.count())
                if n == 0:
                    return "Collection enquete_scores_commune vide."
                res = col.query(
                    query_embeddings=[q_emb],
                    n_results=n,
                    include=["documents", "metadatas"],
                )
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                if not docs:
                    return "Aucun score trouvé pour cette requête."
                lines = [
                    f"[{meta.get('commune', '?')} — {meta.get('n_repondants', '?')} répondants]\n{doc[:600]}"
                    for doc, meta in zip(docs, metas)
                ]
                return "\n\n".join(lines)
            except Exception as e:
                return f"Erreur enquete_scores_search : {e}"

        elif name == "decompose":
            try:
                from rag_v10_raptor_subq import decompose_question
                question = params["question"]
                n = params.get("n", 3)
                sub_questions = decompose_question(question, n=n)
                return json.dumps({"sub_questions": sub_questions}, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"Erreur decompose : {e}"

        return f"Outil inconnu : {name}"

    # --------------------------------------------------------
    # Boucle ReAct
    # --------------------------------------------------------

    def _run_react_loop(self, question: str, max_iterations: int = MAX_ITERATIONS) -> dict:
        """
        Boucle ReAct via Anthropic Tool Use natif.
        Retourne {answer, context, tool_calls, iterations}.
        """
        messages = [{"role": "user", "content": question}]
        accumulated_context = []
        tool_calls_log = []

        for iteration in range(max_iterations):
            response = self._anthropic.messages.create(
                model=EXECUTOR_MODEL,
                max_tokens=4000,
                system=_SYSTEM_EXECUTOR,
                tools=TOOLS,
                messages=messages,
            )

            # Reponse finale (pas d'appel d'outil)
            if response.stop_reason == "end_turn":
                final_answer = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return {
                    "answer": final_answer,
                    "context": "\n\n".join(accumulated_context),
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                }

            # Traiter les appels d'outils
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_params = block.input
                params_short = json.dumps(tool_params, ensure_ascii=False)[:120]
                print(f"  [v11 it.{iteration+1}] {tool_name}({params_short})")

                # Executer l'outil
                result = self._execute_tool(tool_name, tool_params)

                # CRAG gate
                crag = self._crag_score(question, [result] if result else [])
                print(f"  [v11 it.{iteration+1}] CRAG score: {crag:.3f}")

                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "params": tool_params,
                    "crag_score": round(crag, 3),
                })

                accumulated_context.append(f"[{tool_name}]\n{result[:2000]}")

                # Signaler un contexte peu pertinent au LLM via le contenu
                if crag < 0.2 and result and not result.startswith("Erreur"):
                    content = f"[Résultat peu pertinent (score={crag:.2f}), essaie un autre outil ou reformule.]\n{result[:600]}"
                else:
                    content = result

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

            messages.append({"role": "user", "content": tool_results})

        # Max iterations atteint : forcer la generation
        print(f"  [v11] Max iterations ({max_iterations}) atteint, generation forcee")
        force_resp = self._anthropic.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=2000,
            system=(
                _SYSTEM_EXECUTOR
                + "\n\nATTENTION : le nombre maximum d'itérations est atteint. "
                "Génère la meilleure réponse possible avec le contexte déjà disponible."
            ),
            messages=messages,
        )
        final_answer = next(
            (b.text for b in force_resp.content if hasattr(b, "text")), ""
        )
        return {
            "answer": final_answer,
            "context": "\n\n".join(accumulated_context),
            "tool_calls": tool_calls_log,
            "iterations": max_iterations,
        }

    # --------------------------------------------------------
    # Fast path (questions simples → v9 direct + Mistral Small)
    # --------------------------------------------------------

    def _call_mistral_fast(self, question: str, context: str) -> str:
        """Génération rapide via Mistral Small pour le fast path."""
        from openai import OpenAI
        client = OpenAI(api_key=self._mistral_key, base_url=MISTRAL_BASE_URL)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=FAST_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Tu es un assistant spécialisé dans la qualité de vie en Corse. "
                                "Réponds en français, uniquement à partir du contexte fourni. "
                                "Si le contexte ne contient pas l'information, dis-le clairement."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Contexte :\n{context[:4000]}\n\nQuestion : {question}",
                        },
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(2 ** attempt * 2)
                else:
                    raise

    # --------------------------------------------------------
    # Point d'entree principal
    # --------------------------------------------------------

    def query(
        self,
        question: str,
        k: int = 7,
        max_iterations: int = MAX_ITERATIONS,
        use_fast_path: bool = True,
    ) -> Tuple[str, List[Dict], Optional[List[Dict]]]:
        """
        Pose une question au pipeline agentique.

        Args:
            question: Question en français.
            k: Nombre de documents pour le fast path et les outils (défaut 7).
            max_iterations: Nombre max d'itérations ReAct (défaut 5).
            use_fast_path: Si True, les questions simples passent par v9 direct.

        Returns:
            (answer, sources_list, sub_questions_list)
            - sources_list : List[Dict] avec clés rank, source_type, crag_score, extrait
            - sub_questions_list : List[Dict] avec clés sub_question, sub_answer, score (ou None)
        """
        classification = self._classify(question)
        print(f"  [v11] Classification: {classification['complexity']}")

        # Hors domaine
        if classification["complexity"] == "hors_domaine":
            return (
                "Je suis spécialisé dans l'analyse de la qualité de vie en Corse "
                "à partir de verbatims citoyens et d'indicateurs territoriaux. "
                "Je ne suis pas en mesure de répondre à cette question.",
                [],
                None,
            )

        # Fast path : question simple → v9 direct
        if use_fast_path and classification["complexity"] == "simple":
            print("  [v11] Fast path → v9 direct + Mistral Small")
            context_str, raptor_sources = self._raptor.query(question, k=k)

            # Enrichissement OppChoVec si pertinent
            if any(kw in question.lower() for kw in _OPPCHOVEC_KEYWORDS):
                opp_docs = self._raptor.query_oppchovec(question, k=3)
                if opp_docs:
                    context_str += "\n\n[Scores OppChoVec (objectif/quanti)]\n" + "\n\n".join(
                        d["text"] for d in opp_docs
                    )
                    raptor_sources += [
                        {
                            "rank": len(raptor_sources) + i,
                            "source_type": "oppchovec_score",
                            "extrait": d["text"][:1500],
                        }
                        for i, d in enumerate(opp_docs)
                    ]

            answer = self._call_mistral_fast(question, context_str)
            return answer, raptor_sources, None

        # Boucle ReAct pour questions complexes
        print(f"  [v11] Boucle ReAct (max {max_iterations} itérations)")
        result = self._run_react_loop(question, max_iterations=max_iterations)

        # Formater les sources depuis le log des tool_calls
        sources = [
            {
                "rank": i,
                "source_type": f"tool_{tc['tool']}",
                "crag_score": tc["crag_score"],
                "iteration": tc["iteration"],
                "extrait": tc.get("params", {}).get("query", tc.get("params", {}).get("commune", ""))[:200],
            }
            for i, tc in enumerate(result["tool_calls"])
        ]

        # Formater sub_questions (affichage dans le frontend, même format que v10)
        sub_questions = [
            {
                "sub_question": (
                    f"[Itération {tc['iteration']}] {tc['tool']}"
                    f"({json.dumps(tc['params'], ensure_ascii=False)[:80]})"
                ),
                "sub_answer": f"CRAG score: {tc['crag_score']:.3f}",
                "score": tc["crag_score"],
            }
            for tc in result["tool_calls"]
        ]

        return result["answer"], sources, sub_questions


# ============================================================
# CLI pour tests standalone
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG v11 — Agentic ReAct + CRAG")
    parser.add_argument("--query", "-q", required=True, help="Question à poser")
    parser.add_argument("--k", type=int, default=7, help="Nombre de documents (défaut 7)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Nb max itérations (défaut 5)")
    parser.add_argument("--no-fast-path", action="store_true", help="Désactiver le fast path")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RAG v11 — Agentic RAG (ReAct + CRAG gate)")
    print("=" * 60)

    pipeline = AgenticRAGPipeline()
    pipeline.init()

    print(f"\nQuestion : {args.query}\n")
    answer, sources, sub_questions = pipeline.query(
        question=args.query,
        k=args.k,
        max_iterations=args.max_iterations,
        use_fast_path=not args.no_fast_path,
    )

    print("\n" + "=" * 60)
    print("RÉPONSE")
    print("=" * 60)
    print(answer)

    if sub_questions:
        print("\n" + "=" * 60)
        print("TRACE DES OUTILS UTILISÉS")
        print("=" * 60)
        for sq in sub_questions:
            print(f"  {sq['sub_question']}")
            print(f"    → {sq['sub_answer']}")

    print(f"\nSources ({len(sources)}) :")
    for s in sources:
        print(f"  [{s.get('source_type','?')}] CRAG={s.get('crag_score','?')} | {s.get('extrait','')[:80]}")
