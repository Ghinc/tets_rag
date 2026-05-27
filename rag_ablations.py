"""
rag_ablations.py — Configurations d'ablation pour l'étude comparative RAG v10.

V_vanilla        : BGE-M3 → top-k chunks bruts → Mistral Large (1 appel, pas de décomposition).
V_decomp         : Décomposition + retrieval brut (hors RAPTOR) + synthèse sans bilan.
V_decomp_raptor  : Géré directement dans api_server via RaptorSubQuestionPipeline(use_bilan=False).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from rag_v10_raptor_subq import (
    RaptorSubQuestionPipeline,
    decompose_question,
    answer_subquestion,
    synthesize_answers,
    _call_mistral,
    CHROMA_PATH,
    DEFAULT_N_SUBQUESTIONS,
)

# Collections RAPTOR à exclure pour les ablations "sans RAPTOR"
_RAPTOR_COLLECTIONS = frozenset({
    "raptor_entretiens_summaries",
    "raptor_summaries",
    "raptor_enquete_summaries",
})

_SYSTEM_VANILLA = (
    "Tu es un expert en analyse de la qualité de vie en Corse. "
    "Tu reçois des extraits de documents (verbatims citoyens, enquêtes, indicateurs territoriaux). "
    "Réponds directement à la question posée en t'appuyant strictement sur les documents fournis. "
    "Si les données sont insuffisantes pour répondre, dis-le clairement. "
    "Ne mentionne pas de communes ou de chiffres absents du contexte. "
    "Ne fabrique jamais de faits. "
    "RÈGLE GÉOGRAPHIQUE : réponds uniquement sur la/les commune(s) mentionnée(s) dans la question. "
    "RÈGLE ATTRIBUTION : pour tout chiffre ou constat, mentionne sa source "
    "(enquête citoyenne, indicateurs territoriaux, entretiens qualitatifs)."
)


def _get_raw_sources(retriever, question: str, k: int) -> Tuple[str, List[Dict]]:
    """
    Retrieval sur collections brutes uniquement (hors RAPTOR).
    Utilisé par VanillaRAG et DecompOnlyRAG.
    Retourne (context_str, sources_list).
    """
    detected = retriever._detect_dimensions(question)

    extra = retriever.query_extra_sources(
        question, k=k,
        communes=detected.get("noms") or None,
        epci=detected.get("epci"),
    )
    raw_extra = [r for r in extra if r.get("collection") not in _RAPTOR_COLLECTIONS]

    opp_parts: List[Dict] = []
    if retriever._oppchovec:
        try:
            opp_raw = retriever.query_oppchovec(
                question, k=2,
                communes=detected.get("noms") or None,
            )
            for r in opp_raw:
                opp_parts.append({
                    "text":       r["text"],
                    "label":      f"Scores OppChoVec — {r['meta'].get('commune', 'Corse')}",
                    "collection": "oppchovec_scores",
                    "meta":       r["meta"],
                })
        except Exception:
            pass

    all_sources = opp_parts + raw_extra

    context = "\n\n".join(
        f"[{s['label']}]\n{s['text'][:1500]}"
        for s in all_sources[:k]
    )
    sources = [
        {
            "content":     s["text"],
            "metadata":    s.get("meta", {}),
            "source_type": s.get("collection", "raw"),
            "label":       s.get("label", ""),
        }
        for s in all_sources[:k]
    ]
    return context, sources


class VanillaRAG:
    """
    V_vanilla : RAG basique en 1 étape.
    BGE-M3 → top-k sur collections brutes (hors RAPTOR) → Mistral Large.
    k est passé à query() — utiliser k=10 (v_vanilla_k10) ou k=25 (v_vanilla_k25).
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self._chroma_path = chroma_path
        self._retriever = None
        self._initialized = False

    def init(self):
        from rag_v9_raptor import RaptorRetriever
        self._retriever = RaptorRetriever(chroma_path=self._chroma_path)
        self._retriever.init()
        self._initialized = True
        print("VanillaRAG initialisé")

    def query(self, question: str, k: int = 10) -> Tuple[str, List[Dict]]:
        assert self._initialized, "Appelez init() d'abord."
        context, sources = _get_raw_sources(self._retriever, question, k)
        answer = _call_mistral(
            f"Contexte :\n{context[:12000]}\n\nQuestion : {question}",
            _SYSTEM_VANILLA,
        )
        return answer, sources


class DecompOnlyRAG:
    """
    V_decomp : décomposition + retrieval brut (hors RAPTOR) + synthèse Mistral sans bilan.
    Identique à V_full sauf : collection RAPTOR exclue du retrieval, bilan absent.
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self._chroma_path = chroma_path
        self._pipeline: Optional[RaptorSubQuestionPipeline] = None
        self._initialized = False

    def init(self):
        self._pipeline = RaptorSubQuestionPipeline(chroma_path=self._chroma_path)
        self._pipeline.init()
        self._initialized = True
        print("DecompOnlyRAG (V_decomp) initialisé")

    def query(self, question: str, k: int = 5,
              n_subquestions: int = DEFAULT_N_SUBQUESTIONS
              ) -> Tuple[str, List[Dict], Dict, List[Dict]]:
        assert self._initialized, "Appelez init() d'abord."
        retriever = self._pipeline.retriever

        print(f"\n[v_decomp] Question : {question}")
        print(f"[v_decomp] Etape 1/3 : Décomposition (Mistral Large)...")
        try:
            sub_questions = decompose_question(question, n=n_subquestions)
        except RuntimeError as _e:
            print(f"[v_decomp] Question hors-domaine ou indécomposable : {_e}")
            _refusal = _call_mistral(
                f"Question : {question}",
                "Tu es un assistant spécialisé en qualité de vie en Corse. "
                "Cette question ne relève pas de ton domaine d'expertise. "
                "Réponds poliment que tu ne peux pas répondre à cette question.",
                max_tokens=300, temperature=0.3,
            )
            return _refusal, [], {}, []

        print(f"[v_decomp] Etape 2/3 : Retrieval brut + Haiku par sous-question...")
        sub_qa_pairs: List[Tuple[str, str]] = []
        all_sources: List[Dict] = []

        for i, sq in enumerate(sub_questions, 1):
            print(f"  [{i}/{len(sub_questions)}] {sq[:80]}...")
            context_str, sources = _get_raw_sources(retriever, sq, k)
            ans = answer_subquestion(sq, context_str)
            sub_qa_pairs.append((sq, ans))
            for s in sources:
                s["sub_question_idx"] = i
                s["sub_question"] = sq
            all_sources.extend(sources)

        print(f"[v_decomp] Etape 3/3 : Synthèse finale sans bilan (Mistral Large)...")
        final_answer = synthesize_answers(
            question, sub_qa_pairs,
            source_bilan=None,
            use_bilan=False,
        )

        sub_qa_list = [
            {"idx": i + 1, "question": sq, "answer": ans}
            for i, (sq, ans) in enumerate(sub_qa_pairs)
        ]
        print("[v_decomp] Pipeline terminé.")
        return final_answer, all_sources, {}, sub_qa_list
